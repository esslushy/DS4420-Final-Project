from argparse import ArgumentParser
import json
from Model.EMPNN import EMPNNModel
from Model.GCNN import GCNNModel
from World.World import World
import importlib
import sys
from pathlib import Path
import torch
from ProjectParameters import MAX_ANGLE_CHANGE, MAX_SPEED_CHANGE, GAMMA
import numpy as np

def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = foo
    spec.loader.exec_module(foo)
    return foo

def main(world: Path, config: dict, output_dir: Path):
    # Get world
    world: World = load_module("custom_env", world).world
    # Intialize actor and critic networks
    if config["model"] == "EMPNN":
        actor = EMPNNModel(outputs=[2, 2])
        critic = EMPNNModel(outputs=[1])
    elif config["model"] == "GCNN":
        actor = GCNNModel(outputs=[2, 2])
        critic = GCNNModel(outputs=[1])
    else:
        raise ValueError(f"No model of type {config['model']}")
    if config["optimizer"] == "SGD":
        actor_optim = torch.optim.SGD(actor.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
        critic_optim = torch.optim.SGD(critic.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
    elif config["optimizer"] == "Adam":
        actor_optim = torch.optim.Adam(actor.parameters(), lr=config["learning_rate"])
        critic_optim = torch.optim.Adam(critic.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError(f"No optimizer of type {config['optimizer']}")
    # Repeat for however many episodes
    rewards = []
    losses = []
    for episode in config["num_episodes"]:
        episode_rewards = list()
        episode_loss = list()
        for step in config["max_steps"]:
            actor_optim.zero_grad()
            critic_optim.zero_grad()
            # Update world
            world.update_entities_position()
            # Get state
            curr_state = world.compute_graph()
            # Sample change to robot
            pred_means, pred_deviations = actor(curr_state)
            speed_change, angle_change = torch.normal(pred_means, pred_deviations)
            speed_change = torch.clip(speed_change, -MAX_SPEED_CHANGE, MAX_SPEED_CHANGE)
            angle_change = torch.clip(angle_change, -MAX_ANGLE_CHANGE, MAX_ANGLE_CHANGE)
            # Apply change to robot
            world.robot.update_velocity(speed_change, angle_change)
            # Update robot position
            world.robot.update_position()
            # Update rest of entities to next state, compute collisions
            world.update_entities_velocity()
            world.compute_collisions()
            # Observe next state
            reward = world.compute_reward(step)
            next_state = world.compute_graph()
            # Get value from critic
            pred_reward_cs = critic(curr_state)[0]
            pred_reward_ns = critic(next_state)[0]
            # Compute loss
            if world.robot_reached_goal():
                advantage = reward - pred_reward_cs
                critic_target = reward
            else:
                advantage = reward + (GAMMA * pred_reward_ns) - pred_reward_cs
                critic_target = reward + (GAMMA * pred_reward_ns)
            actor_loss = torch.mean(-advantage * torch.log(pred_deviations))
            critic_loss = torch.nn.functional.mse_loss(pred_reward_cs, critic_target)
            # Update parameters
            actor_loss.backward()
            critic_loss.backward()
            actor_optim.step()
            critic_optim.step()
            # Save reward and loss
            episode_rewards.append(reward)
            episode_loss.append(actor_loss.item() + critic_loss.item())
        # Reset world at end of episode
        world.reset()
        rewards.append(np.mean(episode_rewards))
        losses.append(np.mean(episode_loss))
    # Save results
    output_dir.mkdir(exist_ok=True)
    config["training_world"] = world
    with output_dir.joinpath("config.json").open("wt+") as f:
        json.dump(config, f)
    with output_dir.joinpath("results.json").open("wt+") as f:
        json.dump({
            "training_loss": losses,
            "training_reward": rewards
        }, f)
    # Save models
    torch.save(actor.state_dict(), output_dir.joinpath("actor.model"))
    torch.save(critic.state_dict(), output_dir.joinpath("critic.model"))

    
if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("world", help="The path to the world to run on.", type=Path)
    args.add_argument("config", help="The config file for the experiment", type=Path)
    args.add_argument("output_dir", help="The location to save the final results to", type=Path)
    args = args.parse_args()
    with args.config.open() as f:
        config = json.load(f)
    main(args.world, config, args.output_dir)