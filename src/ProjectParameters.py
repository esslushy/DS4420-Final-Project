import math

STEP_SIZE=0.5
MAX_ANGLE_CHANGE=math.pi/8 # Radians
MAX_SPEED_CHANGE=0.1
GAMMA = 0.9 # Reward decay
BETA = 0.2 # Entropy factor
BETA_DECAY = 0.01 # How fast Beta decays (encourage less exploration over time)
# Reward
DISTANCE_SCALE = 10000
COLLISION_SCALE = 10
TIME_SCALE = 1
SUCCESS_REWARD = 10000
SPEED_SCALE = 1 # Bonus provided for robot going faster 