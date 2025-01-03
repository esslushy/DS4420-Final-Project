from argparse import ArgumentParser
import json
from Model import EMPNNModel, GTransformerModel, GCNNModel
from World.World import World
from pathlib import Path
import torch
from ProjectParameters import MAX_ANGLE_CHANGE, MAX_SPEED_CHANGE, GAMMA, BETA, BETA_DECAY
import numpy as np
from tqdm import tqdm
from Utils import load_module, log_prob, compute_entropy, Decayer, scale_reward

def main(world_pth: Path, config: dict, output_dir: Path, from_existing: Path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Get world
    world: World = load_module("custom_env", world_pth).world
    # Intialize actor and critic networks
    if config["model"] == "EMPNN":
        actor = EMPNNModel(outputs=[2, 2])
        critic = EMPNNModel(outputs=[1])
    elif config["model"] == "GCNN":
        actor = GCNNModel(outputs=[2, 2])
        critic = GCNNModel(outputs=[1])
    elif config["model"] == "GTransformer":
        actor = GTransformerModel(outputs=[2, 2])
        critic = GTransformerModel(outputs=[1])
    else:
        raise ValueError(f"No model of type {config['model']}")
    # Load prior changed models
    if from_existing:
        actor.load_state_dict(torch.load(from_existing.joinpath("actor.model")))
        critic.load_state_dict(torch.load(from_existing.joinpath("critic.model")))
    actor.to(device)
    critic.to(device)
    if config["optimizer"] == "SGD":
        actor_optim = torch.optim.SGD(actor.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
        critic_optim = torch.optim.SGD(critic.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "Adam":
        actor_optim = torch.optim.Adam(actor.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        critic_optim = torch.optim.Adam(critic.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"No optimizer of type {config['optimizer']}")
    # Schedulers
    actor_scheduler = torch.optim.lr_scheduler.StepLR(actor_optim, 1, 0.9)
    critic_scheduler = torch.optim.lr_scheduler.StepLR(critic_optim, 1, 0.9)
    # Repeat for however many episodes
    rewards = []
    final_rewards = []
    losses = []
    collisions = []
    successful_episodes = []
    final_distances = []
    decayer = Decayer(BETA_DECAY)
    for episode in range(config["num_episodes"]):
        print(f"Episode {episode+1}:")
        episode_rewards = list()
        episode_losses = list()
        for step in tqdm(range(config["max_steps"]), "Step", leave=False):
            actor_optim.zero_grad()
            critic_optim.zero_grad()
            # Update world
            world.update_entities_position()
            # Get state
            curr_state = world.compute_graph().to(device)
            # Sample change to robot
            pred_means, pred_deviations = actor(curr_state)
            pred_means = pred_means.tanh()[0] * MAX_SPEED_CHANGE
            pred_deviations = pred_deviations.relu()[0] + 1e-8
            speed_change, angle_change = torch.normal(pred_means, pred_deviations)
            speed_change = torch.clip(speed_change, -MAX_SPEED_CHANGE, MAX_SPEED_CHANGE)
            angle_change = torch.clip(angle_change, -MAX_ANGLE_CHANGE, MAX_ANGLE_CHANGE)
            # Apply change to robot
            world.robot.update_velocity(speed_change.item(), angle_change.item())
            # Update robot position
            world.robot.update_position(world.width, world.height)
            # Update rest of entities to next state, compute collisions
            world.update_entities_velocity()
            world.compute_collisions()
            # Observe next state
            reward = world.compute_reward(step)
            reward = scale_reward(reward, config, world)
            next_state = world.compute_graph().to(device)
            # Get value from critic
            pred_reward_cs = critic(curr_state)[0].tanh()
            pred_reward_ns = critic(next_state)[0].tanh()
            # Compute loss
            if world.robot_reached_goal():
                advantage = reward - pred_reward_cs
                critic_target = torch.tensor(reward, dtype=torch.float32).reshape(1,1).to(device)
            else:
                advantage = reward + (GAMMA * pred_reward_ns) - pred_reward_cs
                critic_target = reward + (GAMMA * pred_reward_ns)
            actor_loss = -advantage * (log_prob(speed_change, pred_means[0], pred_deviations[0]) \
                                     + log_prob(angle_change, pred_means[1], pred_deviations[1]))
            # Encourage exploration by allowing high changes to angle
            actor_loss = actor_loss - (decayer.decay(BETA) * compute_entropy(pred_deviations[1]))
            critic_loss = torch.nn.functional.mse_loss(pred_reward_cs, critic_target)
            # Update parameters
            actor_loss.backward(retain_graph=True)
            critic_loss.backward()
            actor_optim.step()
            critic_optim.step()
            # Save reward and loss
            episode_rewards.append(reward)
            episode_losses.append(actor_loss.item() + critic_loss.item())
            if world.robot_reached_goal():
                successful_episodes.append(episode)
                # Lower learning rate on each success
                actor_scheduler.step()
                critic_scheduler.step()
                break
        # Reset world at end of episode
        avg_episode_reward = np.mean(episode_rewards)
        avg_episode_loss = np.mean(episode_losses)
        print("Avg. Reward:", avg_episode_reward)
        print("Loss:", avg_episode_loss)
        print("Collisions:", world.robot.num_collisions)
        print("Final Reward:", reward)
        if world.robot_reached_goal():
            print("Reached Goal!")
            final_distance = 0
        else:
            final_distance = world.robot.position.distance(world.goal.position)
            print("Final Distance:", final_distance)
        rewards.append(avg_episode_reward)
        final_rewards.append(reward)
        final_distances.append(final_distance)
        losses.append(avg_episode_loss)
        collisions.append(world.robot.num_collisions)
        world.reset()
        # Decay learning rate
        actor_scheduler.step()
        critic_scheduler.step()
    # Save results
    output_dir.mkdir(exist_ok=True, parents=True)
    config["training_world"] = str(world_pth)
    with output_dir.joinpath("config.json").open("wt+") as f:
        json.dump(config, f, indent=2)
    with output_dir.joinpath("results.json").open("wt+") as f:
        json.dump({
            "training_loss": losses,
            "training_reward": rewards,
            "final_reward": final_rewards,
            "final_distance": final_distances,
            "collisions": collisions,
            "successful_episodes": successful_episodes
        }, f, indent=2)
    # Save models
    torch.save(actor.state_dict(), output_dir.joinpath("actor.model"))
    torch.save(critic.state_dict(), output_dir.joinpath("critic.model"))
    
if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("world", help="The path to the world to run on.", type=Path)
    args.add_argument("config", help="The config file for the experiment", type=Path)
    args.add_argument("output_dir", help="The location to save the final results to", type=Path)
    args.add_argument("--from-existing", help="Train from existing models. Path to the folder", type=Path)
    args = args.parse_args()
    with args.config.open() as f:
        config = json.load(f)
    main(args.world, config, args.output_dir, args.from_existing)