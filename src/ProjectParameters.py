import math

STEP_SIZE=0.5
MAX_ANGLE_CHANGE=math.pi/8 # Radians
MAX_SPEED_CHANGE=0.1
GAMMA = 0.9 # Reward decay
BETA = 0.1 # Entropy factor
BETA_DECAY = 0.05 # How fast Beta decays (encourage less exploration over time)
# Reward
DISTANCE_SCALE = 100
COLLISION_SCALE = 10000
TIME_SCALE = 0.1
SUCCESS_REWARD = 1000
SPEED_SCALE = 0.1 # Bonus provided for robot going faster 

