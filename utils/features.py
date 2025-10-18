import pennylane as qml
import pennylane.numpy as np

from typing import Dict, List
from utils.adjacency import adjacency_with_props

MAX_ATOMS = 16

def permutation_invariant_encoding(
    adj: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    props: Dict,
    prop_weights: np.ndarray
) -> None:
    num_qubits = adj.shape[0]
    layers = betas.shape[0]
    for i in range(num_qubits): qml.Hadamard(wires=i)
    for l in range(layers):
        for i in range(num_qubits):
            for j in range(i):
                weight = adj[i, j]
                if abs(weight) < 1e-6: continue # skip near zero weights
                qml.IsingZZ(weight * gammas[l] * 2, wires=[i,j])
        for i in range(num_qubits):
            qml.RX(adj[i,i] * betas[l], wires=i)
    #properties
    homo_angle = props['homo'] * prop_weights[0]
    lumo_angle = props['lumo'] * prop_weights[1]
    gap_angle = homo_angle - lumo_angle
    zpve_angle = props['zpve'] * prop_weights[2]
    for i in range(num_qubits):
        qml.RY(gap_angle, wires=i)
        qml.RZ(zpve_angle, wires=i)
    mu_scale = 1 + props['mu'] * prop_weights[3]
    alpha_scale = 1 + props['alpha'] * prop_weights[4]
    for i in range(num_qubits-1):
        qml.IsingXX(mu_scale * 0.1, wires=[i, i+1])
        qml.IsingYY(alpha_scale * 0.1, wires=[i, i+1])

device = qml.device('default.qubit', wires=MAX_ATOMS)
@qml.qnode(device, interface='autograd') # use autograd for integration with MLX
def make_circuit(
    adj: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    props: Dict,
    prop_weights: np.ndarray
) -> np.ndarray:
    permutation_invariant_encoding(adj, betas, gammas, props, prop_weights)
    return [qml.expval(qml.PauliZ(i)) for i in range(adj.shape[0])]