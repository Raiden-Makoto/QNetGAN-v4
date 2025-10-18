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

### 2. GeneratorMLX.py ⏳ **NOT IMPLEMENTED**

**Purpose**: MLX-based generator that creates molecular embeddings from noise.

**Planned Features**:
- Input: Random noise vectors
- Output: Generated molecular embeddings (16-dimensional)
- Framework: MLX (Apple Silicon optimized)
- Integration: Works with FeatureEncoder embeddings

**Architecture**: *To be implemented*

---

### 3. DiscriminatorMLX.py ⏳ **NOT IMPLEMENTED**

**Purpose**: MLX-based discriminator that distinguishes between real and generated molecular embeddings.

**Planned Features**:
- Input: Molecular embeddings (real or generated)
- Output: Probability score (real vs fake)
- Framework: MLX (Apple Silicon optimized)
- Integration: Works with FeatureEncoder and GeneratorMLX outputs

**Architecture**: *To be implemented*