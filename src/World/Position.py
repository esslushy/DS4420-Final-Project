class Position:
    """
    Defines a position in the world.
    """
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def distance(self, other: "Position"):
        """
        Computes the l2 distance between this position and the other position
        """
        return ((self.x-other.x)**2 + (self.y - other.y)**2)**(1/2)