from argparse import ArgumentParser
from World.World import World
from Utils import load_module
import torch
from ProjectParameters import MAX_ANGLE_CHANGE, MAX_SPEED_CHANGE
from pathlib import Path
import json
from Model import EMPNNModel, GTransformerModel, GCNNModel

def main(world: World, model):
    step = 0
    while not world.robot_reached_goal():
        if step > 100000:
            break
        step += 1
        world.update_entities_position()
        curr_state = world.compute_graph()
        # Sample change to robot
        pred_means, pred_deviations = model(curr_state)
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
        world.compute_collisions()
        world.update_entities_velocity()
    return step, world.robot.num_collisions


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("model", help="Directory of the model to use", type=Path)
    args.add_argument("worlds", help="Path to the worlds to test", type=Path)
    args.add_argument("--iters", default=10, type=int, help="The number of iterations to do.")
    args = args.parse_args()
    
    with args.model.joinpath("config.json").open() as f:
        config = json.load(f)
    
    if config["model"] == "EMPNN":
        model = EMPNNModel(outputs=[2, 2])
    elif config["model"] == "GTransformer":
        model = GTransformerModel(outputs=[2, 2])
    elif config["model"] == "GCNN":
        model = GCNNModel(outputs=[2, 2])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(args.model.joinpath("actor.model"), map_location=device, weights_only=True))

    results = dict()

    for w in args.worlds.glob("*.py"):
        print("Running: ",w.stem)
        results[w.stem] = dict()
        results[w.stem]["steps"] = list()
        results[w.stem]["collisions"] = list()
        for i in range(args.iters):
            m = load_module("custom_env", w)
            steps, collisions = main(m.world, model)
            results[w.stem]["steps"].append(steps)
            results[w.stem]["collisions"].append(collisions)

    with args.model.joinpath("test_results.json").open("wt+") as f:
        json.dump(results, f, indent=2)