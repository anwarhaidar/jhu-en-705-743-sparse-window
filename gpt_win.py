import torch
import math

from gpt import TransformerDecoderBlock, GPTModel


class CustomWindowMHA(torch.nn.Module):
    """
    Multi-head attention with sliding window + dilated sparse pattern.
    Each token attends to:
      1. The previous `window_size` tokens (sliding window)
      2. Tokens at dilated positions (every `dilation` tokens before the window)

    Mask is cached once up to `max_seq_len` and then sliced each forward.
    """

    def __init__(self, d_model, n_heads, window_size=128, dilation=4, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.dilation = dilation
        self.max_seq_len = max_seq_len

        # qkv and output projection – same pattern as CustomMHA in gpt.py
        self.qkv = torch.nn.Parameter(
            0.01 * torch.randn((3 * d_model, d_model))
        )
        self.wo = torch.nn.Parameter(
            0.01 * torch.randn((d_model, d_model))
        )

        # Precompute and cache the sparse mask on CPU; move to device as needed.
        self.register_buffer("_cached_mask", self._create_sparse_mask(max_seq_len))


    def _create_sparse_mask(self, max_seq_len: int) -> torch.Tensor:
        """
        Build a (max_seq_len, max_seq_len) boolean mask where True means "allowed to attend".
        """
        ms = max_seq_len
        mask = torch.zeros((ms, ms), dtype=torch.bool)

        # Causal lower triangle
        causal = torch.tril(torch.ones((ms, ms), dtype=torch.bool))

        for i in range(ms):
            # 1) Sliding window: last `window_size` positions including i
            w_start = max(0, i - self.window_size + 1)
            mask[i, w_start : i + 1] = True

            # 2) Dilated positions before the window
            pos = i - self.window_size - self.dilation
            while pos >= 0:
                mask[i, pos] = True
                pos -= self.dilation

        # Enforce causality
        mask = mask & causal
        return mask

    def get_attention_mask(self, seq_len: int, device) -> torch.Tensor:
        """
        Returns a (1, S, S) mask for the current sequence length on the right device.
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len} "
                "used to build the cached mask."
            )
        mask = self._cached_mask[:seq_len, :seq_len]  # (S, S)
        return mask.unsqueeze(0).to(device)           # (1, S, S)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) or (S, D)
        added_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            added_batch = True

        B, S, D = x.shape
        dh = D // self.n_heads
        device = x.device

        # Project to Q, K, V
        QKV = x @ self.qkv.T                     # (B, S, 3D)
        Q, K, V = torch.chunk(QKV, 3, dim=-1)    # (B, S, D) each

        def split_heads(t):
            # (B, S, D) -> (B * h, S, dh)
            t = t.view(B, S, self.n_heads, dh)
            t = t.transpose(1, 2)                # (B, h, S, dh)
            return t.reshape(B * self.n_heads, S, dh)

        q_heads = split_heads(Q)
        k_heads = split_heads(K)
        v_heads = split_heads(V)

        # Scaled dot-product attention
        k_t = k_heads.transpose(1, 2)            # (B*h, dh, S)
        scores = torch.matmul(q_heads, k_t) / math.sqrt(float(dh))  # (B*h, S, S)

        # Apply sparse causal mask
        mask = self.get_attention_mask(S, device)  # (1, S, S)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.nn.functional.softmax(scores, dim=-1)
        x = torch.matmul(attn, v_heads)         # (B*h, S, dh)

        # Merge heads back
        x = x.view(B, self.n_heads, S, dh)
        x = x.transpose(1, 2).reshape(B, S, D)  # (B, S, D)

        # Final output projection
        x = x @ self.wo.T                       # (B, S, D)

        if added_batch:
            x = x[0]

        return x


class TransformerDecoderBlockWindow(TransformerDecoderBlock):
    """
    Drop-in replacement for TransformerDecoderBlock that swaps its MHA
    for CustomWindowMHA, reusing everything else from gpt.py.
    """

    def __init__(self, d_model, n_heads, window_size=128, dilation=4, max_seq_len=512):
        super().__init__(d_model, n_heads)
        # Override the original mha with our windowed version
        self.mha = CustomWindowMHA(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            dilation=dilation,
            max_seq_len=max_seq_len,
        )


class GPTWindowModel(GPTModel):
    """
    GPT model that uses windowed attention blocks instead of the default ones.
    Reuses GPTModel's embeddings, forward pass, and output layer as-is.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        layers,
        vocab_size,
        max_seq_len,
        window_size=128,
        dilation=4,
    ):
        # Initialize the vanilla GPTModel (creates embeddings, layers, fc_out)
        super().__init__(d_model, n_heads, layers, vocab_size, max_seq_len)

        # Now replace the stack of blocks with windowed versions.
        # We assume GPTModel defines self.layers as a ModuleList of TransformerDecoderBlock.
        self.layers = torch.nn.ModuleList(
            [
                TransformerDecoderBlockWindow(
                    d_model=d_model,
                    n_heads=n_heads,
                    window_size=window_size,
                    dilation=dilation,
                    max_seq_len=max_seq_len,
                )
                for _ in range(layers)
            ]
        )
        # Everything else (word_embeddings, position_embeddings, fc_out, and forward)
        # is reused from GPTModel.


if __name__ == "__main__":
    # Sanity test
    print("Testing GPTWindowModel...")
    model = GPTWindowModel(
        d_model=128,
        n_heads=8,
        layers=4,
        vocab_size=1000,
        max_seq_len=512,
        window_size=64,
        dilation=4,
    )

    B = 8
    S = 256
    x = torch.randint(0, 1000, (B, S))
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Test passed!")
