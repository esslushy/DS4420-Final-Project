from World.Entity import Entity
from World.Vector2D import Vector2D
from typing import List

class GoalEntity(Entity):
    """
    Represents an entity that goes between goals
    """
    def __init__(self, radius: float, speed: float, goals: List[Vector2D]):
        """
        Initializes a new entity at the given position with the given velocity and radius.
        Entity oscillates between each goal.

        Args:
            radius: The radius of the entity.
            position: The initial position of the entity.
            velocity: The velocity of the entity.
            goals: The goals for the entity to traverse
        """
        self.speed = speed
        self.current_goal = 1
        self.goals = goals
        super().__init__(radius, goals[0], self.compute_velocity(goals[0], goals[1]))

    def compute_velocity(self, current_pos: Vector2D, next_pos: Vector2D):
        direction = current_pos.displace(next_pos)
        direction /= direction.magnitude()
        return direction * self.speed
    
    def update_velocity(self, world_width, world_height):
        # Check if at goal pos, if it is, adjust velocity to next goal
        if self.position.distance(self.goals[self.current_goal]) < self.radius:
            self.current_goal = (self.current_goal + 1) % len(self.goals)
            self.velocity = self.compute_velocity(self.position, self.goals[self.current_goal])

    def reset(self):
        super().reset()
        self.current_goal = 1