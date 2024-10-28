from World.World import World
from World.StillEntity import StillEntity
from World.Vector2D import Vector2D
from World.Robot import Robot
world = World(
    50, 50,
    StillEntity(2.5, Vector2D(25, 47.5)),
    Robot(2.5, 3, 3),
    [StillEntity(5, Vector2D(25, 25))]
)