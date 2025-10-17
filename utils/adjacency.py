import numpy as np
import networkx as nx
import warnings

from typing import Tuple, Optional, Dict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.linalg import fractional_matrix_power

from utils.constants import (
    MU_MEAN,
    MU_STD,
    ALPHA_MEAN,
    ALPHA_STD,
    HOMO_MEAN,
    HOMO_STD,
    LUMO_MEAN,
    LUMO_STD,
    ZPVE_MEAN,
    ZPVE_STD,
)

def pad_adjacency_matrix(adj: np.ndarray, target_size: int = 16) -> Optional[np.ndarray]:
    current_size = adj.shape[0]
    if current_size > target_size: return None
    if current_size == target_size: return adj
    padded = np.zeros((target_size, target_size))
    padded[:current_size, :current_size] = adj
    return padded

def normalized_adjacency_matrix(smiles: str, pad_to: int = 16) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        warnings.warn(f"Could not convert {smiles} to a molecule.")
        return None
    mol = Chem.AddHs(mol)
    adj = Chem.GetAdjacencyMatrix(mol, useBO=True)
    graph = nx.from_numpy_array(adj)
    lap = nx.laplacian_matrix(graph)
    deg = lap + adj # degree matrix = laplacian + adjacency
    deg_inv = fractional_matrix_power(deg, -0.5)
    normalized = deg_inv @ adj @ deg_inv
    padded = pad_adjacency_matrix(normalized, pad_to)
    return padded

def adjacency_with_props(smiles: str, props: Dict, pad_to: int = 16) -> Tuple[Optional[np.ndarray], Dict]:
    adj = normalized_adjacency_matrix(smiles, pad_to)
    if adj is None:
        return None, {}
    
    # Add molecular properties here
    props['mu'] = (props['mu'] - MU_MEAN) / MU_STD
    props['alpha'] = (props['alpha'] - ALPHA_MEAN) / ALPHA_STD
    props['homo'] = (props['homo'] - HOMO_MEAN) / HOMO_STD
    props['lumo'] = (props['lumo'] - LUMO_MEAN) / LUMO_STD
    props['zpve'] = (props['zpve'] - ZPVE_MEAN) / ZPVE_STD
    
    return adj, props