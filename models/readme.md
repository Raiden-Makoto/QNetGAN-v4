# QNetGANv4 Model Components

This directory contains the core model components for the QNetGAN (Quantum Molecular Generative Adversarial Network) architecture.

## Model Architecture Overview

QNetGANv4 uses a hybrid approach combining:
- **Quantum circuits** (PennyLane) for molecular feature encoding
- **Classical neural networks** (MLX) for generation and discrimination
- **Framework conversion** between PennyLane and MLX for seamless integration

## Model Components

### 1. FeatureEncoder.py ✅ **IMPLEMENTED**

**Purpose**: Quantum feature encoder that converts molecular SMILES strings into quantum embeddings.

**Key Features**:
- Uses PennyLane quantum circuits for molecular encoding
- Incorporates molecular properties (HOMO, LUMO, gap, etc.) into quantum gates
- Generates 16-dimensional embeddings from adjacency matrices
- Supports framework conversion (PennyLane ↔ MLX)
- Trainable quantum circuit parameters (`betas`, `gammas`, `prop_weights`)

**Architecture**:
- Input: SMILES strings + molecular properties
- Quantum circuit: Permutation-invariant encoding with molecular properties
- Output: 16-dimensional quantum embeddings
- Framework: PennyLane (with MLX conversion support)

**Usage**:
```python
encoder = FeatureEncoder(n_layers=2)
embeddings = encoder(smiles, props)  # PennyLane numpy
mlx_embeddings = encoder.pennylane_to_mlx(embeddings)  # Convert to MLX
```

---

### 2. GeneratorMLX.py ✅ **IMPLEMENTED**

**Purpose**: MLX-based generator that creates molecular graphs from quantum embeddings.

**Key Features**:
- Takes quantum embeddings from FeatureEncoder as input
- Generates both node types and edge types for molecular graphs
- Uses mixed graph neural network architecture (GCNConv + GATv2Conv)
- Supports batch processing with proper tensor reshaping
- Framework: MLX (Apple Silicon optimized)

**Architecture**:
- Input: Quantum embeddings `(batch_size, 16)` from FeatureEncoder
- Projection: Linear layer to hidden dimension `(batch_size, 64)`
- Graph Processing: GCNConv → GCNConv → GATv2Conv → GCNConv
- Node Prediction: MLP for atom types `(batch_size, 16, 5)`
- Edge Prediction: MLP for bond types `(batch_size, 16, 16, 4)`
- Residual connections and attention mechanisms

**Usage**:
```python
generator = GeneratorMLX(
    latent_dim=16,      # Match FeatureEncoder output
    hidden_dim=64,
    num_layers=4,
    num_heads=4,
    num_nodes=16,
    num_atom_types=5,
    num_bond_types=4
)
node_logits, edge_logits = generator(quantum_embeddings)
```

---

### 3. DiscriminatorMLX.py ✅ **IMPLEMENTED**

**Purpose**: MLX-based discriminator that distinguishes between real and generated molecular graphs.

**Key Features**:
- Takes node features and adjacency matrices as input
- Converts adjacency matrices to edge_index format for GCNConv
- Uses mixed graph neural network architecture (GCNConv + GATv2Conv)
- Supports batch processing with different graph structures
- Framework: MLX (Apple Silicon optimized)

**Architecture**:
- Input: Node features `(batch_size, 16, 5)` and adjacency matrix `(batch_size, 16, 16)`
- Edge Index Conversion: Adjacency matrix → edge_index format
- Graph Processing: GCNConv → GCNConv → GATv2Conv
- Graph Pooling: Mean aggregation across nodes
- Classification: MLP for real/fake prediction `(batch_size, 1)`

**Usage**:
```python
discriminator = DiscriminatorMLX(
    num_heads=4,
    num_atom_types=5,
    num_bond_types=4,
    latent_dim=64,
    hidden_dim=128
)
rf_logits = discriminator(node_features, adjacency_matrix)
```