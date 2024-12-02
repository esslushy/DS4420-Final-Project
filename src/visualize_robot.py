from argparse import ArgumentParser
from World.World import World
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from functools import partial
from Utils import load_module
import torch
from ProjectParameters import MAX_ANGLE_CHANGE, MAX_SPEED_CHANGE
from pathlib import Path
import json
from Model import EMPNNModel, GTransformerModel, GCNNModel

def draw_world(frame, world: World, model, ax):
    if world.robot_reached_goal():
        return
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
    ax.clear()
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    for entity in world.entities:
        ax.add_patch(plt.Circle((entity.position.x, entity.position.y), entity.radius, color="black"))
    ax.add_patch(plt.Circle((world.robot.position.x, world.robot.position.y), world.robot.radius, color="red"))
    ax.add_patch(plt.Circle((world.goal.position.x, world.goal.position.y), world.goal.radius, color="green"))
    world.update_entities_velocity()

def main(world: World, steps: int, save: Path, model):
    fig, ax = plt.subplots(figsize=(8,8), tight_layout=True)
    ani = animation.FuncAnimation(fig, partial(draw_world, world=world, model=model, ax=ax), steps, interval=20, repeat=True)
    if save:
        writer = animation.PillowWriter(fps=20,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save(save, writer=writer)
    else:
        plt.show()

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("model", help="Directory of the model to use", type=Path)
    args.add_argument("world", help="Path to the world")
    args.add_argument("--save", help="The path to save the file to.", type=Path)
    args.add_argument("--steps", type=int, help="The number of steps to simulate the world", default=10000)
    args = args.parse_args()
    m = load_module("custom_env", args.world)

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

    main(m.world, args.steps, args.save, model)