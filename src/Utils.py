import importlib
import sys
import torch
import math
from ProjectParameters import SUCCESS_REWARD, COLLISION_SCALE, DISTANCE_SCALE, TIME_SCALE, SPEED_SCALE
from World.World import World

def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = foo
    spec.loader.exec_module(foo)
    return foo

def log_prob(value, mean, stdev):
    """
    Computes the gaussian log prob
    """
    return -torch.log(stdev) - ((1/2)*math.log(math.pi/2)) - ((1/2)*(((value-mean)/stdev)**2))

def compute_entropy(stdev):
    return 0.5 * (1 + torch.log(2 * torch.pi * (stdev**2)))

def scale_reward(reward, config, world: World):
    """
    Scales reward down to [-1, 1] range
    """
    reward_max = SUCCESS_REWARD + SPEED_SCALE
    reward_min = -(COLLISION_SCALE * config["max_steps"]) \
        - (DISTANCE_SCALE * ((world.width**2) + (world.height**2))) \
        - (TIME_SCALE * config["max_steps"])
    return (2 * ((reward - reward_min)/(reward_max - reward_min))) - 1

class Decayer():
    """
    Decays a value
    """
    def __init__(self, decay_rate):
        self.decay_rate = decay_rate
        self.step = -1

    def decay(self, value):
        self.step += 1
        return value * math.exp(-self.decay_rate * self.step)