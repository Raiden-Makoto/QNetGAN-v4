import numpy as np
import networkx as nx
import warnings
from rdkit import Chem
from scipy.linalg import fractional_matrix_power

def normalized_adjacency_matrix(smiles: str) -> np.ndarray:
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
    return normalized
