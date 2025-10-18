import mlx.core as mx
import mlx.nn as nn
from mlx_graphs.nn import GCNConv, GATv2Conv, Linear

class DiscriminatorMLX(nn.Module):
    def __init__(
        self,
        num_heads: int=4,
        num_atom_types: int=5,
        num_bond_types: int=4,
        latent_dim: int=64,
        hidden_dim: int=128,
        dropout_rate: float=0.1,
    ):
        super().__init__()
        self.conv_block = [
            GCNConv(num_atom_types, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        ]
        self.mlp = nn.Sequential(
            Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            Linear(latent_dim, 1)
        )

    def __call__(self, node_feats, adj):
        """
        node_feats: (B, N, A) one-hot atom features
        adj: (B, N, N) integer bond-type adjacency
        Returns: real/fake logits (B, 1)
        """
        batch_size, num_nodes, _ = node_feats.shape
        
        # Convert adjacency matrix to edge_index format
        edge_indices = []
        for b in range(batch_size):
            edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adj[b, i, j] > 0:  # If there's a bond
                        edges.append([i, j])
            if edges:
                edge_indices.append(mx.array(edges).T)
            else:
                # No edges case - create self-loops
                edge_indices.append(mx.array([[i, i] for i in range(num_nodes)]).T)
        
        h = node_feats
        for i, conv_layer in enumerate(self.conv_block):
            # Process each sample in batch
            h_batch = []
            for b in range(batch_size):
                h_reshaped = h[b].reshape(-1, h.shape[-1])  # (num_nodes, features)
                h_out = conv_layer(edge_indices[b], h_reshaped)  # (num_nodes, features)
                h_batch.append(h_out)
            h = mx.stack(h_batch)  # (batch_size, num_nodes, features)
        
        h_graph = mx.mean(h, axis=1)  # (batch_size, features)
        rf_logits = self.mlp(h_graph)  # (batch_size, 1)
        return rf_logits
