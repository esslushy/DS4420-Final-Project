from abc import ABC
from World.Vector2D import Vector2D
from ProjectParameters import STEP_SIZE

class Entity(ABC):
    """
    Represents an entity in the world.
    """
    def __init__(self, radius: float, position: Vector2D, velocity: Vector2D):
        self.radius = radius
        self.position = position
        self.velocity = velocity
        self.initial_position = Vector2D(position.x, position.y)
        self.initial_velocity = Vector2D(velocity.x, velocity.y)

    def update_position(self):
        """
        Updates position based on velocity
        """
        self.position += self.velocity * STEP_SIZE

    def update_velocity(self):
        """
        Updates velocity
        """
        raise NotImplementedError("Velocity updates must be implemented in base class")
    
    def collided_with(self, other: "Entity"):
        return self.position.distance(other.position) < (self.radius + other.radius)
    
    def reset(self):
        """
        Resets the entity to the original position and momentum
        """
        self.position = Vector2D(self.initial_position.x, self.initial_position.y)
        self.velocity = Vector2D(self.initial_velocity.x, self.initial_velocity.y)