from torch import nn
from Model.Layers import EquivariantMessageLayer, EquivariantUpdateLayer, GatedEquivariantBlock
from Model.CutOff import CosineCutoff
from Model.GaussianDistance import GaussianDistance
import torch

class EMPNNModel(nn.Module):
    def __init__(self, outputs: list, relevant_radius=100.0, h_dim=128, num_interation_layers=3, num_fully_connected_layers=1) -> None:
        super().__init__()
        self.gauss_expansion = GaussianDistance(0, relevant_radius, 1)
        
        self.rbf_dim = self.gauss_expansion.filter.shape[0]
        self.h_dim = h_dim

        self.embed_layer = GatedEquivariantBlock(1, 1, h_dim, h_dim, h_dim)

        self.message_passing_layers = []
        self.update_layers = []

        for _ in range(num_interation_layers):
            self.message_passing_layers.append(EquivariantMessageLayer(h_dim, self.rbf_dim, CosineCutoff(relevant_radius)))
            self.update_layers.append(EquivariantUpdateLayer(h_dim))

        self.message_passing_layers = nn.ModuleList(self.message_passing_layers)
        self.update_layers = nn.ModuleList(self.update_layers)

        self.outputs = []

        for output_shape in outputs:
            linear_layers = list()
            for _ in range(num_fully_connected_layers):
                linear_layers.append(nn.Linear(h_dim, h_dim))
                linear_layers.append(nn.SiLU())
            self.outputs.append(nn.Sequential(*linear_layers, nn.Linear(h_dim, output_shape)))

        self.outputs = nn.ModuleList(self.outputs)

    def forward(self, data):
        # Vector values of nodes
        v = data.node_attr.reshape(-1, 2, 1)
        # Scalar Values
        s = data.x

        # Expand dims
        v, s = self.embed_layer(v, s)
        
        d_ij = torch.norm(data.edge_attr, dim=-1, keepdim=True)
        d_ij[d_ij == 0] = 1 # Stability
        dir_ij = data.edge_attr / d_ij
        phi_ij = self.gauss_expansion.expand(d_ij)

        for message, update in zip(self.message_passing_layers, self.update_layers):
            v, s = message(v, s, data.edge_index, phi_ij, d_ij, dir_ij)
            v, s = update(v, s)

        s = s[0].reshape(1, -1)
        
        # Assemble all node data batchwise in a sum
        return [output(s) for output in self.outputs]