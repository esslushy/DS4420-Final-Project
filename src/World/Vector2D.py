import numpy as np
from typing import Union

class Vector2D():
    """
    Represents a 2 dimensional vector
    """
    def __init__(self, x: float, y: float):
        """
        Makes a 2 dimensional vector from x and y values
        """
        self.x = x
        self.y = y

    @staticmethod
    def from_magnitude_direction(magnitude: float, direction: float):
        """
        Makes a Vector2D from magnitude and direction.
        """
        return Vector2D(magnitude * np.cos(direction), magnitude * np.sin(direction))

    def magnitude(self):
        """
        Returns the l2 magnitude of this vector.
        """
        return np.sqrt(self.x**2 + self.y**2)
    
    def distance(self, other: "Vector2D"):
        """
        Gives the mangitudal distance between this and another vector
        """
        return np.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)
    
    def displace(self, other: "Vector2D"):
        """
        Gives the displacement vector that when added to this vector
        gives the other vector.
        """
        return Vector2D(other.x - self.x, other.y - self.y)

    def as_array(self):
        """
        Returns this vector2D as an array
        """
        return [self.x, self.y]
    
    def __add__(self, other: Union["Vector2D", int, float]):
        if type(other) == Vector2D:
            return Vector2D(self.x + other.x, self.y + other.y)
        elif type(other) == int or type(other) == float:
            return Vector2D(self.x + other, self.y + other)
        else:
            raise TypeError("Can't do addition on a non-number or Vector2D")
        
    def __mul__(self, other: Union[int, float]):
        if type(other) == int or type(other) == float:
            return Vector2D(self.x * other, self.y * other)
        else:
            raise TypeError("Can't do multiplication on a non-number")