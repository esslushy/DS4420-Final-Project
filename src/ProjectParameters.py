import math

STEP_SIZE=0.5
MAX_ANGLE_CHANGE=math.pi/8 # Radians
MAX_SPEED_CHANGE=0.1
GAMMA = 0.9 # Reward decay
# Reward
DISTANCE_SCALE = 1
COLLISION_SCALE = 10
TIME_SCALE = 10
SUCCESS_REWARD = 100
SPEED_SCALE = 1 # Bonus provided for robot going faster 