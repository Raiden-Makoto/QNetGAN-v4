from pennylane import numpy as np
from typing import List, Dict, Optional, Callable
import numpy as numpy_np  # Regular numpy for conversions

from utils.features import make_circuit
from utils.adjacency import adjacency_with_props

import mlx.core as mx
import pennylane as qml

class FeatureEncoder():
    def __init__(self, n_layers: int=1):
        self.betas = np.random.randn(n_layers, requires_grad=True)
        self.gammas = np.random.randn(n_layers, requires_grad=True)
        self.prop_weights = np.random.uniform(0, 1, 5, requires_grad=True)

    def __call__(self, smiles: List[str], props: List[Dict]) -> np.ndarray:
        """
        Encode molecules into embeddings.
        
        Args:
            smiles: List of SMILES strings
            props: List of property dictionaries
        
        Returns:
            Embeddings as PennyLane numpy array (for gradients)
        """
        embeddings = []
        for smile, prop in zip(smiles, props):
            adj, props = adjacency_with_props(smile, prop)
            if adj is None: continue
            embedding = make_circuit(adj, self.betas, self.gammas, props, self.prop_weights)
            embeddings.append(embedding)
        
        # Return empty array if no valid embeddings
        if not embeddings: 
            return np.array([])
        
        embeddings = np.stack(embeddings)
        return embeddings
    
    def get_trainable_params(self) -> List[np.ndarray]:
        """Get trainable parameters for optimization."""
        return [self.betas, self.gammas, self.prop_weights]
    
    def set_params(self, params: List[np.ndarray]):
        """Set parameters from optimization."""
        self.betas, self.gammas, self.prop_weights = params

    @staticmethod
    def pennylane_to_mlx(pennylane_array: np.ndarray) -> mx.array:
        """Convert PennyLane numpy array to MLX array."""
        return mx.array(numpy_np.array(pennylane_array))
    
    @staticmethod
    def mlx_to_pennylane(mlx_array: mx.array) -> np.ndarray:
        """Convert MLX array to PennyLane numpy array."""
        return np.array(numpy_np.array(mlx_array))
