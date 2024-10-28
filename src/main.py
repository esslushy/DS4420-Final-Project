from argparse import ArgumentParser
import json
from Model.EMPNN import EMPNNModel
from Model.GCNN import GCNNModel
from World.World import World
from World.Robot import Robot
import importlib
import sys
from pathlib import Path

def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = foo
    spec.loader.exec_module(foo)
    return foo

def main(world, config):
    # Get world
    world = load_module("custom_env", world).world
    # Intialize actor and critic networks
    if config["model"] == "EMPNN":
        actor = EMPNNModel(outputs=[2, 2])
        critic = EMPNNModel(outputs=[1])
    elif config["model"] == "GCNN":
        actor = GCNNModel(outputs=[2, 2])
        critic = GCNNModel(outputs=[1])
    else:
        raise ValueError(f"No model of type {config['model']}")
    # Repeat for however many episodes
    rewards = []
    loss = []
    for episode in config["num_episodes"]:
        average_reward = 0
        average_loss = 0
        for step in config["max_steps"]:
            # Update world
            # Get state
            # Sample change to robot
            # Apply change to robot
            # Update robot position
            # Update rest of entities
            # Observe next state
            # Compute loss
            # Update parameters
            pass
        # Reset world at end of episode
        world.reset()
    # Save results

    # Save models

    
if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("world", help="The path to the world to run on.", type=Path)
    args.add_argument("config", help="The config file for the experiment", type=Path)
    args = args.parse_args()
    with args.config.open() as f:
        config = json.load(f)
    main(args.world, config)