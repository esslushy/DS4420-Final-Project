import numpy as np

class Vector2D():
    """
    Represents a 2 dimensional vector
    """
    def __init__(self, x: int, y: int):
        """
        Makes a 2 dimensional vector from x and y values
        """
        self.x = x
        self.y = y

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

