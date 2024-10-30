from World.Entity import Entity
from World.Vector2D import Vector2D
from random import random

class WanderEntity(Entity):
    """
    Represents an entity that wanders around the map
    """
    def __init__(self, radius, position, speed):
        super().__init__(radius, position, Vector2D(0, 0))
        self.goal = position
        self.speed = speed

    def reset(self):
        super().reset()
        self.goal = self.position
    
    def compute_velocity(self, current_pos: Vector2D, next_pos: Vector2D):
        direction = current_pos.displace(next_pos)
        direction /= direction.magnitude()
        return direction * self.speed
    
    def update_velocity(self, world_width, world_height):
        # Check if at goal pos, if it is, find next goal and adjust velocity to next goal
        if self.position.distance(self.goal) < self.radius:
            # Choose new goal
            self.goal = Vector2D(
                random() * (world_width - (self.radius * 2)) + self.radius,
                random() * (world_height - (self.radius * 2)) + self.radius
            )
            # Go toward it
            self.velocity = self.compute_velocity(self.position, self.goal)

