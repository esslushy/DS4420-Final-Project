from argparse import ArgumentParser
from World.World import World
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from functools import partial
from Utils import load_module

def draw_world(frame, world: World, ax):
    world.update_entities_position()
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    for entity in world.entities:
        ax.add_patch(plt.Circle((entity.position.x, entity.position.y), entity.radius, color="black"))
    ax.add_patch(plt.Circle((world.robot.position.x, world.robot.position.y), world.robot.radius, color="red"))
    ax.add_patch(plt.Circle((world.goal.position.x, world.goal.position.y), world.goal.radius, color="green"))
    world.update_entities_velocity()

def main(world: World):
    fig, ax = plt.subplots(figsize=(8,8), tight_layout=True)
    ani = animation.FuncAnimation(fig, partial(draw_world, world=world, ax=ax), 10000, interval=20, repeat=True)
    plt.show()

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("world", help="Path to the world")
    args = args.parse_args()
    m = load_module("custom_env", args.world)
    main(m.world)