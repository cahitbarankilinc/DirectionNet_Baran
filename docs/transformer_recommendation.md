# Transformer Integration Proposal for DirectionNet

## Design Goals
- **Joint reasoning over \(v_x, v_y, v_z, t\)** to mitigate incoherent predictions under wide-baseline, low-overlap regimes.
- **Global contextual aggregation** so that directional heads leverage the shared Siamese embedding rather than learning in isolation.
- **Drop-in compatibility** with the existing TensorFlow 1.x-style Keras codebase and 64×64 spherical decoder.

## Architectural Placement
1. **Shared embedding extraction**: keep the current Siamese encoder and decoder blocks unchanged up to the point where `util.distributions_to_directions` produces the unnormalized expectation vectors.
2. **Token formation**:
   - Stack the raw expectation vectors for \(v_x, v_y, v_z, t\) into a tensor of shape `[B, 4, 3]`.
   - Append a learned *context token* derived from the encoder’s 1024-dimension bottleneck (projected to 3 dims) to give the transformer access to the scene-level summary, yielding `[B, 5, 3]` tokens.
3. **Transformer block**:
   - Apply a lightweight two-layer Transformer encoder with:
     - Hidden size 128, 4 attention heads, rotary positional encoding over the token index (simple learned embeddings suffice since sequence length is tiny).
     - Pre-normalization residual layout (LayerNorm → MultiHeadSelfAttention → residual, LayerNorm → MLP → residual) for stability.
     - Feed-forward network dimension 256 with SiLU/Swish activation.
   - Implement attention via `tf.keras.layers.MultiHeadAttention` (available through `tensorflow.compat.v1.keras.layers` in TF ≥2.4 even under v1 compatibility) or a custom projection if running on older releases.
4. **Projection back to directions**:
   - Discard the context token and pass each of the four updated tokens through a shared linear projection to 3D.
   - L2-normalize each vector to enforce unit length before downstream orthogonalization (`svd_orthogonalize`, `gram_schmidt`) or translation usage.

## Training Strategy
- **Initialization**: Start with pre-trained DirectionNet weights; initialize transformer weights with Xavier/Glorot.
- **Loss coupling**: Keep existing direction, distribution, and spread losses but add a consistency term encouraging orthogonality between \(v_x, v_y, v_z\) after the transformer (e.g., sum of squared dot products) and alignment between transformed translation and epipolar geometry derived from rotation (optional).
- **Curriculum**: Freeze the transformer for the first few epochs to stabilize training, then fine-tune end-to-end with a reduced learning rate (e.g., 1e-4 for the transformer, 5e-4 for CNN layers).
- **Practical setup**: Use `--transformer_lr=<value>` to set the smaller refinement learning rate and, when resuming from a backbone checkpoint, pass `--freeze_backbone=True` so that only transformer parameters update during the warm-up stage.
- **Regularization**: Apply dropout (rate 0.1) within the transformer MLP and attention to prevent overfitting to dataset-specific baselines.

## Expected Benefits
- **Contextual coherence**: Self-attention forces each directional head to explain itself relative to the others and to the global token, reducing contradictory outputs.
- **Robust translation**: Translation token attends to rotation axes, mitigating drift when overlap is low and translation cues are weak.
- **Scalability**: The block operates on only five tokens, so the FLOP and memory cost is negligible compared to the spherical decoder; it can be repeated (depth 2–3) if stronger coupling is needed without destabilizing training.

## Implementation Notes
- Wrap the transformer inside a new Keras `Model` (e.g., `DirectionalContextTransformer`) so it can be optionally enabled via a flag.
- Keep the API surface: return both the refined unit vectors and the pre-normalized expectations so downstream code (losses, logging) remains compatible.
- Provide a fallback path that bypasses the transformer when disabled to maintain original benchmarks.

## Future Extensions
- **Cross-attention with image tokens**: Flatten mid-level feature maps (e.g., decoder_block3 output) into a small set of learned queries for finer geometric reasoning.
- **Temporal fusion**: Extend the token list with motion priors when multiple frame pairs are available, using the same transformer backbone.
- **Uncertainty modeling**: Augment tokens with concentration scalars and predict them through the transformer to modulate Von Mises–Fisher dispersion.
