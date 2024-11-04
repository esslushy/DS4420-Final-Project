from argparse import ArgumentParser
import json
from Model.EMPNN import EMPNNModel
from Model.GCNN import GCNNModel
from Model.GTransformer import GTransformer
from World.World import World
from pathlib import Path
import torch
from ProjectParameters import MAX_ANGLE_CHANGE, MAX_SPEED_CHANGE, GAMMA, BETA, BETA_DECAY
import numpy as np
from tqdm import tqdm
from Utils import load_module, log_prob, compute_entropy, scale_reward, Decayer

def main(world_pth: Path, config: dict, output_dir: Path, from_existing: Path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Get world
    world: World = load_module("custom_env", world_pth).world
    # Intialize policy network
    if config["model"] == "EMPNN":
        policy = EMPNNModel(outputs=[2, 2])
    elif config["model"] == "GCNN":
        policy = GCNNModel(outputs=[2, 2])
    elif config["model"] == "GTransformer":
        policy = GTransformer(outputs=[2, 2])
    else:
        raise ValueError(f"No model of type {config['model']}")
    # Load prior changed models
    if from_existing:
        policy.load_state_dict(torch.load(from_existing.joinpath("policy.model")))
    policy.to(device)
    if config["optimizer"] == "SGD":
        policy_optim = torch.optim.SGD(policy.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "Adam":
        policy_optim = torch.optim.Adam(policy.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"No optimizer of type {config['optimizer']}")
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
        episode_log_probs = list()
        episode_entropy = list()
        for step in tqdm(range(config["max_steps"]), "Step", leave=False):
            # Update world
            world.update_entities_position()
            # Get state
            curr_state = world.compute_graph().to(device)
            # Sample change to robot
            pred_means, pred_deviations = policy(curr_state)
            pred_means = torch.nn.functional.tanh(pred_means)[0] * MAX_SPEED_CHANGE
            pred_deviations = torch.nn.functional.relu(pred_deviations)[0] + 1e-8
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
            # Observe reward
            reward = world.compute_reward(step)
            reward = scale_reward(reward, config, world)
            # Save reward and log prob
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob(speed_change, pred_means[0], pred_deviations[0]) \
                                     + log_prob(angle_change, pred_means[1], pred_deviations[1]))
            episode_entropy.append(compute_entropy(pred_deviations[1]))
            if world.robot_reached_goal():
                successful_episodes.append(episode)
                break
        # Update model
        loss = 0
        policy_optim.zero_grad()
        for t in range(step+1):
            add = 0
            for tau in range(t, step+1):
                add = add + (GAMMA**(tau - t))*episode_rewards[tau]
            add = add * episode_log_probs[t] - (decayer.decay(BETA) * episode_entropy[t])
            loss = loss + add
        loss = -(1/(step + 1)) * loss
        loss.backward()
        policy_optim.step()
        # Reset world at end of episode
        avg_episode_reward = np.mean(episode_rewards)
        print("Avg. Reward:", avg_episode_reward)
        print("Loss:", loss.item())
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
        losses.append(loss.item())
        collisions.append(world.robot.num_collisions)
        world.reset()
    # Save results
    output_dir.mkdir(exist_ok=True)
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
    torch.save(policy.state_dict(), output_dir.joinpath("policy.model"))
    
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