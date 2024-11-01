import torch
from torch import nn

class EquivariantUpdateLayer(nn.Module):
    def __init__(self, h_dim, epsilon: float = 1e-8):
        super().__init__()
        self.h_dim = h_dim
        self.epsilon = epsilon

        self.vector_expansion = nn.Linear(h_dim, 2*h_dim, bias=False)
        self.scalar_net = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 3 * h_dim)
        )

    def forward(self, v, s):
        """
          Runs the update layer

          Args:
            v: The vector values for the atoms. These are mutated by the offset vectors dir_ij and phi_ij.
                They are in the shape [number of nodes]x2x[width of embedding or h_dim]
            s: The embedded scalar values of the atoms. These are in the shape [number of nodes]x[width of embedding]
        """
        # Changes v to a [number of nodes]x2x[2 * h_dim]
        v_expanded = self.vector_expansion(v)
        # Splits both into [number of nodes]x2x[h_dim]
        v_u, v_v = torch.split(v_expanded, self.h_dim, dim=-1)
        # Takes norm along 3 layer turning it into [number of nodes]x[h_dim]
        v_v_norm = torch.norm(v_v, dim=-2) + self.epsilon
        # Stacks into [number of nodes]x[2 * h_dim]
        s_stack = torch.cat([s, v_v_norm], dim=-1)
        # Runs the scalar net. Outputs [number of nodes]x[3 * h_dim]
        s_scaled = self.scalar_net(s_stack)
        # Splits into 3 copies of [number of nodes]x[h_dim]
        a_vv, a_sv, a_ss = torch.split(s_scaled, self.h_dim, dim=-1)
        # Update vector values. Retains [number of nodes]x2x[width of embedding or h_dim] shape.
        dv = v_u * a_vv[:, None]
        # Scalar update
        ds = a_ss + (torch.sum(v_u * v_v, dim=-2) * a_sv)

        return v + dv, s + ds