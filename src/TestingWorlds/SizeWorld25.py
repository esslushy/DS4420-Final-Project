from World.World import World
from World.Entity import WanderEntity, StillEntity
from World.Vector2D import Vector2D
from World.Robot import Robot
world = World(
    25, 25,
    StillEntity(2.5, Vector2D(22.5, 22.5)),
    Robot(2.5, Vector2D(3, 3)),
    [WanderEntity(2, Vector2D(10, 10), 1.25), WanderEntity(2, Vector2D(15, 15), 1.25)]
)