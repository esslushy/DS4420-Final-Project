import torch
from torch import nn

class GatedEquivariantBlock(nn.Module):
    def __init__(self, v_in, s_in, v_out, s_out, h_dim, epsilon=1e-8) -> None:
        """
          Initializes a gated equivariant block. Used for equivariant transform of feature vectors

          Args:
            v_in: The input features of the vector inputs
            s_in: The input features of the scalar inputs
            v_out: The output features for the vector outputs
            s_out: The output features for the scalar output
            h_dim: The number of nodes in the hidden layer in the scalar vector mixing layer.
        """
        super().__init__()
        self.epsilon = epsilon
        self.v_out = v_out
        self.s_out = s_out

        self.vector_net = nn.Linear(v_in, 2 * v_out, bias=False)
        self.scalar_net = nn.Sequential(
            nn.Linear(v_out + s_in, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, v_out + s_out)
        )

    def forward(self, v, s):
        """
          Runs the gated equivariant block

          Args:
            v: The vector input in shape [number of nodes]x2x[v_in]
            s: The scalar input in shape of [number of nodes]x[s_in]
        """
        # v_mix becomes [number of nodes]x2x[2*v_out]. Applies unbiased linear layer.
        v_mix = self.vector_net(v)
        # Each become [number of nodes]x2x[v_out]
        v_out, v_s = torch.split(v_mix, self.v_out, dim=-1)
        # Transforms the vector into a scalar value with [number of nodes]x[v_out]
        v_norm = torch.norm(v_s, dim=-2) + self.epsilon
        # Stacks scalar vector values to scalar values to form [number of nodes]x[s_in + v_out]
        s_stack = torch.cat((v_norm, s), dim=-1)
        # Applies network with bias become [number of nodes]x[s_out + v_out]
        s_out_mix = self.scalar_net(s_stack)
        # Splits them into [number of nodes]x[s_out] and [number of nodes]x[v_out]
        s_out, v_s_out = torch.split(s_out_mix, [self.s_out, self.v_out], dim=-1)
        # Have to unsqueeze second dimension of v_s_out so it can multiply properly. Multiplicatively affects v_out
        return v_out * v_s_out[:, None], s_out