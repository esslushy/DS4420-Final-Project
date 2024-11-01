import torch

class GaussianDistance(torch.nn.Module):
    """
    Expands the distance by Gaussian basis.
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
            Minimum distance
        dmax: float
            Maximum distance
        step: float
            Step size for the Gaussian filter
        """
        super().__init__()
        assert dmin < dmax
        assert dmax - dmin > step
        self.register_buffer("filter", torch.arange(dmin, dmax + step, step))
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
            A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
            Expanded distance matrix with the last dimension of length
            len(self.filter)
        """
        return torch.exp(-(distances.reshape(-1, 1) - self.get_buffer("filter")) ** 2 /
                      self.var ** 2)