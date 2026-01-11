## Decoder transformer recommendation

Add a lightweight transformer block after `decoder_block3` output. Tokenize the
feature map as `H*W` tokens, apply self-attention plus a small MLP, then reshape
back to `[B, H, W, C]` before continuing the decoder.
