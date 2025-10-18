import mlx.core as mx
import mlx.nn as nn

# Max valence per atom type: [H, C, N, O, F]
MAX_VALENCE = mx.array([1, 4, 3, 2, 1])

def compute_valence_mask(node_feats, edge_logits):
    """
    Returns boolean mask (B, N, N) indicating valid edges under valence constraints.
    """
    B, N, A = node_feats.shape
    atom_type_idx = mx.argmax(node_feats, axis=-1)  # (B, N)
    atom_max_val = MAX_VALENCE[atom_type_idx]  # (B, N)
    
    bond_type_idx = mx.argmax(edge_logits, axis=-1)  # (B, N, N)
    bond_orders = mx.where(bond_type_idx > 0, bond_type_idx, 0)
    total_bonds = mx.sum(bond_orders, axis=2)  # (B, N)
    
    # Broadcast to pairwise
    tb_i = mx.expand_dims(total_bonds, 2)
    tb_i = mx.broadcast_to(tb_i, (B, N, N))
    tb_j = mx.expand_dims(total_bonds, 1)
    tb_j = mx.broadcast_to(tb_j, (B, N, N))
    mv_i = mx.expand_dims(atom_max_val, 2)
    mv_i = mx.broadcast_to(mv_i, (B, N, N))
    mv_j = mx.expand_dims(atom_max_val, 1)
    mv_j = mx.broadcast_to(mv_j, (B, N, N))
    
    valid_i = (tb_i + bond_orders) <= mv_i
    valid_j = (tb_j + bond_orders) <= mv_j
    return mx.logical_and(valid_i, valid_j)

def gumbel_softmax_sample(logits, tau, hard=True):
    """Gumbel-Softmax with optional straight-through estimator."""
    gumbel_noise = -mx.log(-mx.log(mx.random.uniform(shape=logits.shape) + 1e-10) + 1e-10)
    y = mx.softmax((logits + gumbel_noise) / tau, axis=-1)
    if hard:
        # Manual one-hot encoding using broadcasting since mlx doesn't have one-hot
        max_indices = mx.argmax(y, axis=-1, keepdims=True)
        y_hard = mx.where(mx.arange(y.shape[-1]) == max_indices, 1.0, 0.0)
        y = mx.stop_gradient(y_hard - y) + y
    return y

def sample_graph(node_logits, edge_logits, tau):
    """
    Convert logits to discrete graph with valence enforcement.
    Returns node_feats (B, N, A), adj (B, N, N)
    """
    # Sample nodes
    node_feats = gumbel_softmax_sample(node_logits, tau, hard=True)
    
    # Sample edges
    edge_probs = gumbel_softmax_sample(edge_logits, tau, hard=True)
    edge_indices = mx.argmax(edge_probs, axis=-1)
    
    # Compute valence mask
    valence_mask = compute_valence_mask(node_feats, edge_logits)
    
    # Enforce mask: set invalid bonds to 0 (no bond)
    adj = mx.where(valence_mask, edge_indices, 0)
    return node_feats, adj