from World.World import World
from World.Entity import WanderEntity, StillEntity
from World.Vector2D import Vector2D
from World.Robot import Robot
world = World(
    75, 75,
    StillEntity(2.5, Vector2D(72.5, 72.5)),
    Robot(2.5, Vector2D(3, 3)),
    [WanderEntity(2, Vector2D(27.5, 27.5), 1.25), WanderEntity(2, Vector2D(40, 40), 1.25)]
)