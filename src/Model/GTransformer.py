from torch_geometric.nn.conv import TransformerConv
import torch.nn as nn
import torch

class GTransformerModel(nn.Module):
    """
    Represents a graph transformer network
    """
    def __init__(self, outputs: list, node_dim: int = 64, edge_dim: int = 32, num_conv_layers: int = 4,
                  h_dim: int = 128, num_fully_connected_layers: int = 6) -> None:
        """
        Initializes a GTransformer model.

        Args:
            outputs: A list of all outputs for the model.
            node_dim: The node embedding dim.
            edge_dim: The edge embedding dim.
            num_conv_layers: The number of graph interaction layers.
            h_dim: The hidden dimensionality of the model.
            num_fully_connected_layers: The number of feed forward layers.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_conv_layers
        self.h_dim = h_dim

        self.node_embedding = nn.Linear(4, node_dim)
        self.edge_embedding = nn.Linear(2, edge_dim)

        self.conv_layers = nn.ModuleList(
            [TransformerConv(in_channels=node_dim, out_channels=node_dim, edge_dim=edge_dim) 
             for _ in range(num_conv_layers)]
        )
        self.fc = nn.Linear(node_dim, h_dim)
        self.softplus = nn.Softplus()

        self.outputs = []

        for output_shape in outputs:
            linear_layers = list()
            for _ in range(num_fully_connected_layers):
                linear_layers.append(nn.Linear(h_dim, h_dim))
                linear_layers.append(nn.Softplus())
            self.outputs.append(nn.Sequential(*linear_layers, nn.Linear(h_dim, output_shape)))

        self.outputs = nn.ModuleList(self.outputs)

    def forward(self, data):
        x = torch.hstack([data.x, data.node_attr])
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(data.edge_attr)

        for layer in self.conv_layers:
            x = layer(x, data.edge_index, edge_attr)

        # Select only the goal node
        x = x[0].reshape(1, -1)
        x = self.softplus(x)
        x = self.fc(x)
        x = self.softplus(x)
        return [output(x) for output in self.outputs]
    
