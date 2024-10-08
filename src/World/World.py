from World.Position import Position
from World.Entity import Entity
from typing import List

class World():
    def __init__(self, width: int, height: int, goal: Position, robot: Entity, entities: List[Entity]) -> None:
        self.width = width
        self.height = height
        self.goal = goal

        self.entities = entities

    def step(self, step: int):
        # Update positions by entity position + small_step * velocity.
        # Don't allow position update that will lead to collision for our entities

        # robot will be updated last, allowed to collide for learning purposes

        # Compute Collisions

        # Run update velocity for each entity based on collisions, cu
        for e in self.entities:
            e.update(collision[e]) #
        # update robot with goal, colision, and step num
        self.robot.update(self.goal, collision[self.robot], step)
        pass