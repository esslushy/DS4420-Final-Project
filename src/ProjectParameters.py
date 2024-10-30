import math

STEP_SIZE=0.5
MAX_ANGLE_CHANGE=math.pi/8 # Radians
MAX_SPEED_CHANGE=0.1
GAMMA = 0.99 # Reward decay
# Reward
DISTANCE_SCALE = 10
COLLISION_SCALE = 1e-1
TIME_SCALE = 1e-1
SUCCESS_REWARD = 1000
SPEED_SCALE = 10 # Bonus provided for robot going faster 