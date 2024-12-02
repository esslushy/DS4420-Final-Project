from World.World import World
from World.Entity import StillEntity
from World.Vector2D import Vector2D
from World.Robot import Robot
world = World(
    50, 50,
    StillEntity(2.5, Vector2D(47.5, 25)),
    Robot(2.5, Vector2D(3, 25)),
    [StillEntity(5, Vector2D(25, 25))]
)