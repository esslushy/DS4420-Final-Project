from World.World import World
from World.Entity import StillEntity, CircleEntity
from World.Vector2D import Vector2D
from World.Robot import Robot
import math
world = World(
    50, 50,
    StillEntity(2.5, Vector2D(47.5, 47.5)),
    Robot(2.5, Vector2D(3, 3)),
    [CircleEntity(2, Vector2D(30, 25), Vector2D(0, 1), math.pi/32)]
)