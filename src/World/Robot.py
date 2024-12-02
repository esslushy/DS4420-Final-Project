from World.Entity import Entity
from World.Vector2D import Vector2D

class Robot(Entity):
    """
    Represents the robot in the world
    """
    def __init__(self, radius: float, starting_position: Vector2D):
        """
        Initializes the robot at the given position.

        Args:
            radius: The radius of the robot
            starting_position: The initial location for the robot.
        """
        super().__init__(radius, starting_position, Vector2D(0, 0))
        self.num_collisions = 0

    def update_velocity(self, magnitude, direction):
        """
        Updates the velocity given a specific magnitude and direction change.

        Args:
            magnitude: The change in magnitude.
            direction: The change in direction.
        """
        self.velocity += Vector2D.from_magnitude_direction(magnitude, direction)
        if self.velocity.magnitude() > 1:
            self.velocity /= self.velocity.magnitude()
    
    def reset(self):
        super().reset()
        self.num_collisions = 0