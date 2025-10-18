from pennylane import numpy as np
from typing import List, Dict

from utils.features import make_circuit
from utils.adjacency import adjacency_with_props

import mlx.core as mx

class FeatureEncoder():
    def __init__(self, n_layers: int=1):
        self.betas = np.random.randn(n_layers)
        self.gammas = np.random.randn(n_layers)
        self.prop_weights = np.random.uniform(0, 1, 5)

    def __call__(self, smiles: List[str], props: List[Dict]) -> mx.array:
        embeddings = []
        for smile, prop in zip(smiles, props):
            adj, props = adjacency_with_props(smile, prop)
            if adj is None: continue
            embedding = make_circuit(adj, self.betas, self.gammas, props, self.prop_weights)
            embeddings.append(embedding)
        # Return empty array if no valid embeddings
        if not embeddings: return mx.array([])
        embeddings = np.stack(embeddings)
        return mx.array(embeddings)
        