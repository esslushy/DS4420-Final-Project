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

    def update_entities_position(self):
        for entity in self.entities:
            entity.update_position(self.width, self.height)

    def compute_collisions(self):
        """
        Cound how many entities the robot has collided with.
        Includes walls
        """
        for e in self.entities:
            if self.robot.collided_with(e):
                self.robot.num_collisions += 1
        # Check if hitting wall
        if (self.robot.position.x <= self.robot.radius or \
            self.robot.position.y <= self.robot.radius or \
            self.robot.position.x >= self.width - self.robot.radius or \
            self.robot.position.y >= self.height - self.robot.radius):
            self.robot.num_collisions += 1
    
    def update_entities_velocity(self):
        # Run update velocity for each entity based on collisions
        for e in self.entities:
            e.update_velocity(self.width, self.height)

    def compute_reward(self, step: int):
        """
        Computes the reward for the robot at the current time step
        """    
        return - self.robot.position.distance(self.goal.position) - step \
            - self.robot.num_collisions + (100 if self.robot_reached_goal() else 0)
    
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
        # Size of objects in scene
        sizes = [
            self.goal.radius, 
            self.robot.radius, 
            *[e.radius for e in self.entities],
            0,
            0,
            0,
            0    
        ]

        # Edges are robot(1) to goal(0), entities and walls(2+) to robot(1)
        edge_index = [[1, *range(2, len(velocities))], 
                      [0] + [1] * (len(velocities) - 2)]
        
        # Edge values are displacements
        displacements = [
            self.robot.position.displace(self.goal.position).as_array(),
            *[e.position.displace(self.robot.position).as_array() for e in self.entities],
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
            x=torch.tensor(sizes, dtype=torch.float32),
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

    def robot_reached_goal(self) -> bool:
        return self.robot.collided_with(self.goal)