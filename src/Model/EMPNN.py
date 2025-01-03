from torch import nn
from Model.Layers import EquivariantMessageLayer, EquivariantUpdateLayer, GatedEquivariantBlock
import torch

class EMPNNModel(nn.Module):
    """
    An equivariant message passing graph neural network
    """
    def __init__(self, outputs: list, h_dim: int = 128, num_interation_layers: int = 4, num_fully_connected_layers: int = 6) -> None:
        """
        Initializes a EMPNN model.

        Args:
            outputs: A list of all outputs for the model.
            h_dim: The hidden dimensionality of the model.
            num_interaction_layers: The number of graph interaction layers.
            num_fully_connected_layers: The number of feed forward layers.
        """
        super().__init__()
        
        self.h_dim = h_dim

        self.embed_layer = GatedEquivariantBlock(1, 2, h_dim, h_dim, h_dim)

        self.message_passing_layers = []
        self.update_layers = []

        for _ in range(num_interation_layers):
            self.message_passing_layers.append(EquivariantMessageLayer(h_dim))
            self.update_layers.append(EquivariantUpdateLayer(h_dim))

        self.message_passing_layers = nn.ModuleList(self.message_passing_layers)
        self.update_layers = nn.ModuleList(self.update_layers)

        self.outputs = []

        for output_shape in outputs:
            linear_layers = list()
            for _ in range(num_fully_connected_layers):
                linear_layers.append(nn.Linear(h_dim, h_dim))
                linear_layers.append(nn.Tanh())
            self.outputs.append(nn.Sequential(*linear_layers, nn.Linear(h_dim, output_shape)))

        self.outputs = nn.ModuleList(self.outputs)

    def forward(self, data):
        # Vector values of nodes
        v = data.node_attr.reshape(-1, 2, 1)
        # Scalar Values
        s = data.x

        # Expand dims
        v, s = self.embed_layer(v, s)
        
        d_ij = torch.norm(data.edge_attr, dim=-1, keepdim=True) + 1e-8
        dir_ij = data.edge_attr / d_ij

        for message, update in zip(self.message_passing_layers, self.update_layers):
            v, s = message(v, s, data.edge_index, d_ij, dir_ij)
            v, s = update(v, s)

        s = s[0].reshape(1, -1)
        
        # Assemble all node data batchwise in a sum
        return [output(s) for output in self.outputs]