import torch
from torch import nn
from torch_scatter import scatter

class EquivariantMessageLayer(nn.Module):
    def __init__(self, h_dim, rbf_dim, cutoff_fn):
        """
          Initializes an equivariant message passing layer

          Args:
            h_dim: The hidden dimension used in the model. Same as embedding dim
            rbf_dim: How large the rbf expansion is
            cutoff_fn: What the cutoff function is to use
        """
        super().__init__()

        self.h_dim = h_dim
        
        self.embed_net = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 3*h_dim)
        )

        self.rbf_expansion = nn.Linear(rbf_dim, 3 * h_dim)
        self.cutoff_fn = cutoff_fn

    def forward(self, v, s, edge_index, phi_ij, d_ij, dir_ij):
        """
          Runs a pass of the messaging layer.

          Args:
            v: The vector values for the atoms. These are mutated by the offset vectors dir_ij and phi_ij.
                They are in the shape [number of nodes]x3x[width of embedding or h_dim]
            s: The embedded scalar values of the atoms. These are in the shape [number of nodes]x[width of embedding]
            edge_index: The 2x[number of edges] graph of the ith node to the jth node edge
            phi_ij: This is the RBF of the distances between atoms in the graph. It is in shape [number of edges]x[rbf dim]
            d_ij: This is the exact distances between atoms in the graph. It is in shape [number of edges]x1
            dir_ij: This is the normalized direction vector of the offsets between the atoms in the graph. It is in shape [number of edges]x3

          Returns: v', s' or the new vector values and scalar valuess.
        """
        # currently in shape [number of nodes]x[3 * h_dim]. Holds expanded scalar values
        s_expanded = self.embed_net(s)
        # Expands rbf to [number of edges]x[3 * h_dim]. Cutoff is applied edgewise
        rbf_expanded_cutoff = self.rbf_expansion(phi_ij) * self.cutoff_fn(d_ij) 
        # Index s by edges so it becomes [number of edges]x[3 * h_dim]. Specifically the j edge.
        grouped_values = s_expanded[edge_index[1]] * rbf_expanded_cutoff
        # Split it into the 3 separate change vectors. They are all [number of edges]x[h_dim]
        dv_v, ds, dvr = torch.split(grouped_values, self.h_dim, dim=-1)
        # Update scalar values. We add up all ds with those that have the some source node.
        ds = scatter(ds, edge_index[0], dim=0, dim_size=s.shape[0], reduce="sum")
        """
        Compute the vector update. Note that dvr is expanded to [number of edges]x1x[h_dim] 
        and dir_ij is [number of edges]x3x1, so that when multiplied they match the needed 
        [number of edges]x2x[h_dim] of dv
        """
        dv = dvr[:, None] * dir_ij[..., None] + (dv_v[:, None] * v[edge_index[1]])
        dv = scatter(dv, edge_index[0], dim=0, dim_size=v.shape[0], reduce="sum")

        return v + dv, s + ds