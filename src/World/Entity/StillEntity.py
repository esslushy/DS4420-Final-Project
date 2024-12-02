from World.Entity import Entity
from World.Vector2D import Vector2D

class StillEntity(Entity):
    def __init__(self, radius: float, position: Vector2D):
        """
        Initializes a new entity at the given position with no velocity and the given radius.

        Args:
            radius: The radius of the entity.
            position: The initial position of the entity.
        """
        super().__init__(radius, position, Vector2D(0, 0))

    def update_velocity(self, world_width: float, world_height: float):
        """
        Still entity doesn't move
        """
        return