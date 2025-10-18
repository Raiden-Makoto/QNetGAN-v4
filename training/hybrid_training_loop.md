# hybrid_training_loop.py

Below is the full training loop integrating the Autograd-based quantum encoder and the MLX-based generator and discriminator.

```python
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import mlx.core as mx

# Assume these are defined:
# encoder: Autograd-based QuantumEncoder with .get_params() and .set_params()
# gen: MLX GraphGenerator
# disc: MLX GraphDiscriminator
# opt_q: PennyLane optimizer (e.g., qml.AdamOptimizer)
# opt_cls: MLX optimizer (e.g., mx.optim.Adam) managing gen & disc parameters
# real_data_loader: yields batches of (A_real_mlx, validity_masks_mlx)

for epoch in range(num_epochs):
    for A_real_mlx, validity_mask in real_data_loader:
        batch_size = A_real_mlx.shape[0]

        # 1) Quantum encoding (Autograd interface)
        A_np_batch = A_real_mlx.numpy()  # Convert MLX adjacency → NumPy
        props_dummy = {}  # no conditioning
        z_mlx_list = []
        for A_np in A_np_batch:
            z_mlx_list.append(encoder(A_np, props_dummy))
        z_mlx = mx.stack(z_mlx_list, axis=0)  # shape (B, N)

        # 2) Generate fake graphs in MLX
        fake_atoms, fake_adj = gen(z_mlx)  # MLX tensors

        # 3) Discriminator update
        with mx.autograd.record():
            rf_real, _ = disc(A_real_mlx, validity_mask)
            rf_fake, _ = disc(fake_atoms.detach(), fake_adj.detach())
            loss_D = (
                mx.loss.binary_cross_entropy_with_logits(rf_real, 1)
                + mx.loss.binary_cross_entropy_with_logits(rf_fake, 0)
            )
        loss_D.backward()
        opt_cls.step(disc.parameters())  # update discriminator

        # 4) Generator & Quantum Encoder update
        def generator_step(*q_params):
            # Re-encode with current quantum params
            z_list = []
            for A_np in A_np_batch:
                z_list.append(encoder(A_np, props_dummy))
            z = mx.stack(z_list, axis=0)
            atoms, adj = gen(z)
            rf_fake2, _ = disc(atoms, adj)
            return mx.loss.binary_cross_entropy_with_logits(rf_fake2, 1)

        # 4a) MLX generator update
        with mx.autograd.record():
            loss_G_cls = generator_step()
        loss_G_cls.backward()
        opt_cls.step(gen.parameters())

        # 4b) Quantum encoder update via PennyLane optimizer
        q_params = encoder.get_params()
        q_params = opt_q.step(generator_step, *q_params)
        encoder.set_params(q_params)  # update quantum encoder parameters
```

Save this file as **hybrid_training_loop.py** and import it into your project to keep the training loop available for future iterations.