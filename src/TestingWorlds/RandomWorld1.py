from World.World import World
from World.Entity import WanderEntity, StillEntity
from World.Vector2D import Vector2D
from World.Robot import Robot
world = World(
    50, 50,
    StillEntity(2.5, Vector2D(25, 47.5)),
    Robot(2.5, Vector2D(3, 3)),
    [WanderEntity(2, Vector2D(25, 25), 0.75)]
)