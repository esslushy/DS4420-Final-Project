from World.World import World
from World.Entity import WanderEntity, StillEntity
from World.Vector2D import Vector2D
from World.Robot import Robot
world = World(
    125, 125,
    StillEntity(2.5, Vector2D(122.5, 122.5)),
    Robot(2.5, Vector2D(3, 3)),
    [WanderEntity(2, Vector2D(47.5, 47.5), 1.25), WanderEntity(2, Vector2D(70, 70), 1.25)]
)