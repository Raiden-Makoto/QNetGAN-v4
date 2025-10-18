import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn import GCNConv, GATv2Conv, Linear
from typing import Tuple

class GeneratorMLX(nn.Module):
    def __init__(
        self,
        latent_dim: int=64,
        hidden_dim: int=128,
        num_layers: int=4,
        num_heads: int=4,
        num_nodes: int=16,
        num_atom_types: int=5,
        num_bond_types: int=4,
        dropout_rate: float=0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Project latent vector to hidden dimension
        self.node_proj = Linear(latent_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = [
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False),
            GCNConv(hidden_dim, hidden_dim),
        ]
        
        # Edge prediction MLP
        self.edge_mlp = nn.Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            Linear(hidden_dim, num_bond_types)
        )
        
        # Node prediction MLP
        self.node_mlp = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            Linear(hidden_dim, num_atom_types)
        )

    def __call__(self, z):
        """
        Generate molecular graphs from latent codes.
        
        Args:
            z: (B, latent_dim) quantum latent codes from FeatureEncoder
            
        Returns:
            node_logits: (B, num_nodes, num_atom_types) node type predictions
            edge_logits: (B, num_nodes, num_nodes, num_bond_types) edge type predictions
        """
        batch_size = z.shape[0]
        
        # Project latent codes to hidden dimension
        h = self.node_proj(z)  # (B, hidden_dim)
        
        # Expand to (B, num_nodes, hidden_dim)
        h = mx.expand_dims(h, 1)  # (B, 1, hidden_dim)
        h = mx.broadcast_to(h, (batch_size, self.num_nodes, self.hidden_dim))
        
        # Apply graph convolutions with residual connections
        for conv_layer in self.conv_layers:
            h_residual = h
            # Create edge index for fully connected graph
            edge_index = mx.array([[i, j] for i in range(self.num_nodes) for j in range(self.num_nodes)]).T
            # Reshape h to (num_nodes, hidden_dim) for each batch
            h_reshaped = h.reshape(-1, self.hidden_dim)  # (B*num_nodes, hidden_dim)
            h_out = conv_layer(edge_index, h_reshaped)  # (B*num_nodes, hidden_dim)
            h = h_out.reshape(batch_size, self.num_nodes, self.hidden_dim)
            h = h + h_residual  # Residual connection
            h = nn.ReLU()(h)
        
        # Predict node types
        node_logits = self.node_mlp(h)  # (B, num_nodes, num_atom_types)
        
        # Predict edge types: pairwise concatenation
        h_i = mx.expand_dims(h, 2)  # (B, num_nodes, 1, hidden_dim)
        h_i = mx.broadcast_to(h_i, (batch_size, self.num_nodes, self.num_nodes, self.hidden_dim))
        
        h_j = mx.expand_dims(h, 1)  # (B, 1, num_nodes, hidden_dim)
        h_j = mx.broadcast_to(h_j, (batch_size, self.num_nodes, self.num_nodes, self.hidden_dim))
        
        # Concatenate node features for edge prediction
        edge_input = mx.concatenate([h_i, h_j], axis=-1)  # (B, num_nodes, num_nodes, 2*hidden_dim)
        edge_logits = self.edge_mlp(edge_input)  # (B, num_nodes, num_nodes, num_bond_types)
        
        return node_logits, edge_logits