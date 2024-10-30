from World.World import World
from World.Entity import GoalEntity, StillEntity
from World.Vector2D import Vector2D
from World.Robot import Robot
world = World(
    50, 50,
    StillEntity(2.5, Vector2D(2.5, 47.5)),
    Robot(2.5, Vector2D(47, 3)),
    [GoalEntity(2, 0.5, [Vector2D(5, 5), Vector2D(5, 45), Vector2D(45, 45), Vector2D(45, 5)])]
)