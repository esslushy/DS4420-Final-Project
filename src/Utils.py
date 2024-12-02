import importlib
import sys
import torch
import math
from ProjectParameters import SUCCESS_REWARD, COLLISION_SCALE, DISTANCE_SCALE, TIME_SCALE, SPEED_SCALE, WALL_PENALTY
from World.World import World
from pathlib import Path

def load_module(module_name: str, module_path: Path):
    """
    Loads a python module, used for loading worlds.

    Args:
        module_name: Name for the module.
        module_path: Where the module is located.
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = foo
    spec.loader.exec_module(foo)
    return foo

def log_prob(value, mean: float, stdev: float):
    """
    Computes the gaussian log prob

    Args:
        mean: The gaussian mean.
        stdev: The gaussian standard deviation.
    """
    return -torch.log(stdev) - ((1/2)*math.log(math.pi/2)) - ((1/2)*(((value-mean)/stdev)**2))

def compute_entropy(stdev: float):
    """
    Computes the entropy for a gaussian function

    Args:
        stdev: The gaussian standard deviation.
    """
    return 0.5 * (1 + torch.log(2 * torch.pi * (stdev**2)))

def scale_reward(reward, config: dict, world: World):
    """
    Scales reward down to [-1, 1] range.

    Args:
        config: The configuration of the world to determine max_steps.
        world: The world to get information from.
    """
    reward_max = SUCCESS_REWARD + SPEED_SCALE
    reward_min = -(COLLISION_SCALE * config["max_steps"]) \
        - (DISTANCE_SCALE * ((world.width**2) + (world.height**2))) \
        - (TIME_SCALE * config["max_steps"]) - WALL_PENALTY
    return (2 * ((reward - reward_min)/(reward_max - reward_min))) - 1

class Decayer():
    """
    Decays a value
    """
    def __init__(self, decay_rate: float):
        """
        Initializes a decayer

        Args:
            decay_rate: The amount to decay the reward by each step.
        """
        self.decay_rate = decay_rate
        self.step = -1

    def decay(self, value: float):
        """
        Decay a value by the specific time stpe.

        Args:
            value: The value to decay.
        """
        self.step += 1
        return value * math.exp(-self.decay_rate * self.step)