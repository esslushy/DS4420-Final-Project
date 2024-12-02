from abc import ABC
from World.Vector2D import Vector2D
from ProjectParameters import STEP_SIZE

class Entity(ABC):
    """
    Represents an entity in the world.
    """
    def __init__(self, radius: float, position: Vector2D, velocity: Vector2D):
        """
        Initializes a new entity at the given position with the given velocity and radius.

        Args:
            radius: The radius of the entity.
            position: The initial position of the entity.
            velocity: The velocity of the entity.
        """
        self.radius = radius
        self.position = position
        self.velocity = velocity
        self.initial_position = Vector2D(position.x, position.y)
        self.initial_velocity = Vector2D(velocity.x, velocity.y)

    def update_position(self, world_width: float, world_height: float):
        """
        Updates position based on velocity

        Args:
            world_width: The width of the world to stay within.
            world_height: The height of the world to stay within.
        """
        self.position += self.velocity * STEP_SIZE
        # Bind to world size
        self.position = Vector2D(
            min(max(self.radius, self.position.x), world_width - self.radius),
            min(max(self.radius, self.position.y), world_height - self.radius)
        )

    def update_velocity(self, world_width: float, world_height: float):
        """
        Updates velocity

        Args:
            world_width: The width of the world to stay within.
            world_height: The height of the world to stay within.
        """
        raise NotImplementedError("Velocity updates must be implemented in base class")
    
    def collided_with(self, other: "Entity"):
        """
        Checks if the entity collided with another entity
        """
        return self.position.distance(other.position) < (self.radius + other.radius)
    
    def reset(self):
        """
        Resets the entity to the original position and momentum
        """
        self.position = Vector2D(self.initial_position.x, self.initial_position.y)
        self.velocity = Vector2D(self.initial_velocity.x, self.initial_velocity.y)