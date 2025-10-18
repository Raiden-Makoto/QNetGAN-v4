# QNetGANv4 Utilities

This directory contains utility modules that support the QNetGANv4 molecular generation system.

## Utility Components

### 1. adjacency.py ✅ **IMPLEMENTED**

**Purpose**: Converts molecular SMILES strings into normalized adjacency matrices for quantum circuit processing.

**Key Features**:
- Converts SMILES to RDKit molecular objects
- Generates adjacency matrices with bond order information
- Normalizes adjacency matrices using graph Laplacian
- Pads matrices to 16×16 for consistent input sizes
- Handles invalid SMILES gracefully
- Returns `None` for molecules larger than 16 atoms

**Functions**:
- `normalized_adjacency_matrix(smiles, pad_to=16)`: Main function for adjacency matrix generation
- `adjacency_with_props(smiles, props, pad_to=16)`: Returns adjacency matrix + normalized properties
- `pad_adjacency_matrix(adj, target_size=16)`: Pads matrices to target size

**Usage**:
```python
from utils.adjacency import normalized_adjacency_matrix, adjacency_with_props

# Basic adjacency matrix
adj = normalized_adjacency_matrix('c1ccccc1')  # Benzene: 16×16 matrix

# With properties
adj, props = adjacency_with_props('c1ccccc1', {'mu': 2.5, 'alpha': 80.0, ...})
```

---

### 2. constants.py ✅ **IMPLEMENTED**

**Purpose**: Contains normalization constants calculated from the QM9 dataset for molecular property standardization.

**Key Features**:
- Real statistics from 133,885 molecules in QM9 dataset
- Mean and standard deviation for each property
- Used for normalizing molecular properties before training
- Ensures consistent input scaling across the model

**Properties**:
- **MU_MEAN/STD**: Dipole moment (Debye)
- **ALPHA_MEAN/STD**: Isotropic polarizability (Bohr³)
- **HOMO_MEAN/STD**: Highest occupied molecular orbital (Hartree)
- **LUMO_MEAN/STD**: Lowest unoccupied molecular orbital (Hartree)
- **ZPVE_MEAN/STD**: Zero-point vibrational energy (Hartree)
- **GAP_MEAN/STD**: HOMO-LUMO gap (Hartree)

**Usage**:
```python
from utils.constants import MU_MEAN, MU_STD, HOMO_MEAN, HOMO_STD

# Normalize properties
normalized_mu = (mu_value - MU_MEAN) / MU_STD
normalized_homo = (homo_value - HOMO_MEAN) / HOMO_STD
```

---

### 3. features.py ✅ **IMPLEMENTED**

**Purpose**: Defines the quantum circuit architecture for molecular feature encoding using PennyLane.

**Key Features**:
- Permutation-invariant quantum circuit design
- Incorporates molecular adjacency matrices into quantum gates
- Uses molecular properties to modulate quantum operations
- Supports variable number of layers
- Optimized for 16-qubit circuits (MAX_ATOMS = 16)

**Circuit Architecture**:
1. **Initialization**: Hadamard gates on all qubits
2. **Entangling Layers**: IsingZZ gates based on adjacency matrix weights
3. **Single-Qubit Gates**: RX gates based on diagonal elements
4. **Property Integration**: RY/RZ gates using molecular properties
5. **Scaling Gates**: IsingXX/IsingYY gates with property scaling

**Functions**:
- `permutation_invariant_encoding()`: Core quantum circuit logic
- `make_circuit()`: PennyLane QNode wrapper for the circuit

**Usage**:
```python
from utils.features import make_circuit
import pennylane.numpy as np

# Circuit parameters
betas = np.random.randn(n_layers)
gammas = np.random.randn(n_layers)
prop_weights = np.random.uniform(0, 1, 5)

# Run circuit
embeddings = make_circuit(adj_matrix, betas, gammas, props, prop_weights)
```

---

## Data Flow Integration

### Molecular Processing Pipeline

```
SMILES String
    ↓
RDKit Molecule + Properties
    ↓
Adjacency Matrix (adjacency.py)
    ↓
Normalized Properties (constants.py)
    ↓
Quantum Circuit (features.py)
    ↓
16-Dimensional Embeddings
```