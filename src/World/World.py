from World.Vector2D import Vector2D
from World.Entity import Entity
from World.Robot import Robot
from typing import List
import torch
from torch_geometric.data import Data

class World():
    def __init__(self, width: int, height: int, goal: Entity, robot: Robot, entities: List[Entity]) -> None:
        self.width = width
        self.height = height
        self.goal = goal

        self.entities = entities
        self.robot = robot

    def step(self, step: int):
        # Update positions by entity position + small_step * velocity.
        # Don't allow position update that will lead to collision for our entities
        for entity in self.entities:
            entity.update_position()
        # Let robot update its velocity
        self.robot.update_velocity(self.compute_graph())
        self.robot.update_position()
        # Compute Collisions
        for e in self.entities:
            if self.robot.collided_with(e):
                self.robot.num_collisions += 1
        # Run update velocity for each entity based on collisions
        for e in self.entities:
            e.update_velocity(any([e.collided_with(e_p) for e_p in self.entities])) #
        # Observe reward and next state, update robot graphs
        reward = self.compute_reward(step)
        next_state = self.compute_graph()
        pass

    def compute_reward(self, step: int):
        """
        Computes the reward for the robot at the current time step
        """
        return - self.robot.position.distance(self.goal) - step - self.robot.num_collisions
    
    def compute_graph(self) -> Data:
        """
        Computes the graph of the current world environment
        """
        # Ordering will be goal, robot, entities, walls. Node values are positions
        velocities = [
            self.goal.velocity.as_array(), 
            self.robot.velocity.as_array(), 
            *[e.velocity.as_array() for e in self.entities],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]    
        ]

        # Edges are robot(1) to goal(0), entities and walls(2+) to robot(1)
        edge_index = [[1, *range(2, len(velocities))], 
                      [0] + [1] * (len(velocities) - 2)]
        
        # Edge values are displacements
        displacements = [
            self.robot.position.displace(self.goal.position),
            *[e.position.displace(self.robot.position) for e in self.entities],
            # Left Wall
            [self.robot.position.x, 0],
            # Top Wall
            [0, self.height - self.robot.position.y],
            # Right Wall
            [self.width - self.robot.position.x, 0],
            # Bottom Wall
            [0, self.robot.position.y]
        ]
        return Data(
            edge_index=torch.tensor(edge_index, dtype=torch.int64), 
            node_attr=torch.tensor(velocities, dtype=torch.float32), 
            edge_attr=torch.tensor(displacements, dtype=torch.float32)
        )

    def reset(self):
        """
        Resets the world
        """
        for e in self.entities:
            e.reset()
        self.robot.reset()