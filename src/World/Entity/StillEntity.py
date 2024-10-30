from World.Entity import Entity
from World.Vector2D import Vector2D

class StillEntity(Entity):
    def __init__(self, radius, position: Vector2D):
        super().__init__(radius, position, Vector2D(0, 0))

    def update_velocity(self, world_width: float, world_height: float):
        """
        Still entity doesn't move
        """
        return