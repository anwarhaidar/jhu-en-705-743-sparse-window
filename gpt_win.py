import torch
import math

from linear import CustomLinear
from embedding import CustomEmbedding
from gpt import TransformerDecoderBlock

class CustomWindowMHA(torch.nn.Module):
    """
    Multi-head attention with sliding window + dilated sparse pattern.
    Each token attends to:
    1. The previous w tokens (sliding window)
    2. Tokens at dilated positions (every d tokens before the window)

    OPTIMIZED: Caches mask to avoid recreating it every forward pass.
    """

    def __init__(self, d_model, n_heads, window_size=128, dilation=4, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.dilation = dilation
        self.max_seq_len = max_seq_len

        self.qkv = torch.nn.Parameter(0.01*torch.randn((3*d_model, d_model)))
        self.wo = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

        # Pre-compute and cache the mask for max sequence length
        # This is registered as a buffer so it moves with the model to GPU/CPU
        self.register_buffer('_cached_mask', self._create_sparse_mask(max_seq_len))

    def get_attention_mask(self):
        """Public method to access attention mask for analysis."""
        return self._cached_mask

    def _create_sparse_mask(self, ms):
        """
        Create a causal mask with sliding window + dilated sparse pattern.
        Returns a mask of shape (ms, ms) where mask[i, j] = 1 if token i can attend to token j.
        """
        # Start with zeros
        mask = torch.zeros((ms, ms), dtype=torch.bool)

        # Create causal mask first (lower triangular)
        causal_mask = torch.tril(torch.ones((ms, ms), dtype=torch.bool))

        # For each position, we'll mark which positions to keep
        for i in range(ms):
            # 1. Sliding window: attend to previous w tokens
            window_start = max(0, i - self.window_size + 1)
            mask[i, window_start:i+1] = True

            # 2. Dilated sparse pattern: attend to tokens at positions i-w-d, i-w-2d, etc.
            dilated_pos = i - self.window_size - self.dilation
            while dilated_pos >= 0:
                mask[i, dilated_pos] = True
                dilated_pos -= self.dilation

        # Ensure causality (can't attend to future)
        mask = mask & causal_mask

        return mask

    def forward(self, x):
        added_batch = False
        if len(x.shape) == 2:
            added_batch = True
            x = x[None,:,:]

        # queries, keys, and values
        B, S, D = x.shape
        QKV = x @ self.qkv.T # B, S, 3D
        Q, K, V = torch.chunk(QKV, 3, -1)

        # split into multiple heads
        dh = D//self.n_heads
        q_heads = torch.reshape(Q, (B, S, self.n_heads, dh))
        k_heads = torch.reshape(K, (B, S, self.n_heads, dh))
        v_heads = torch.reshape(V, (B, S, self.n_heads, dh))

        # reshape into (B*h, S, dh) so we isolate sequences for each head
        q_heads = torch.transpose(q_heads, 1, 2).reshape((B*self.n_heads, S, dh))
        k_heads = torch.transpose(k_heads, 1, 2).reshape((B*self.n_heads, S, dh))
        v_heads = torch.transpose(v_heads, 1, 2).reshape((B*self.n_heads, S, dh))

        # Use cached mask (slice if sequence is shorter than max)
        mask = self._cached_mask[:S, :S]
        mask = mask[None, :, :].float()  # Add batch dimension and convert to float

        # attention
        k_heads_t = torch.transpose(k_heads, 1, 2)
        qkt = torch.matmul(q_heads, k_heads_t) / math.sqrt(float(dh))

        # Apply sparse mask efficiently
        # Instead of: qkt = qkt * mask; qkt[qkt == 0] = float('-inf')
        # Use: qkt = qkt.masked_fill(~mask.bool(), float('-inf'))
        qkt = qkt.masked_fill(mask == 0, float('-inf'))

        attn = torch.nn.functional.softmax(qkt, dim=-1)
        x = torch.matmul(attn, v_heads)

        # shmush back into the correct shape
        x = torch.reshape(x, (B, self.n_heads, S, dh))
        x = torch.transpose(x, 1, 2) # B, S, h, dh
        x = torch.reshape(x, (B, S, D))

        # apply projection
        x = x @ self.wo.T

        if added_batch:
            x = x[0]

        return x

class TransformerDecoderBlockWindow(TransformerDecoderBlock):
    """Transformer decoder block with window sparse attention"""
    def __init__(self, d_model, n_heads, window_size=128, dilation=4, max_seq_len=512):
        super().__init__(d_model, n_heads)
        # Override the attention module with window version
        self.mha = CustomWindowMHA(d_model, n_heads, window_size, dilation, max_seq_len)

class GPTWindowModel(torch.nn.Module):
    """
    GPT model with sliding-window + dilated sparse attention.
    Drop-in replacement for GPTModel with additional window_size and dilation parameters.

    OPTIMIZED: Pre-computes attention masks for efficiency.
    """

    def __init__(self, d_model, n_heads, layers, vocab_size, max_seq_len,
                 window_size=128, dilation=4):
        super().__init__()

        self.word_embeddings = CustomEmbedding(vocab_size, d_model)
        self.position_embeddings = CustomEmbedding(max_seq_len, d_model)

        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            block = TransformerDecoderBlockWindow(d_model, n_heads, window_size, dilation, max_seq_len)
            self.layers.append(block)

        self.fc_out = CustomLinear(d_model, vocab_size)

    def forward(self, x):
        B, S = x.shape
        positions = torch.arange(S).to(torch.long).to(x.device)
        positions = positions[None, :]
        positions = positions.repeat(B, 1)

        w_emb = self.word_embeddings(x)
        p_emb = self.position_embeddings(positions)
        x = w_emb + p_emb

        for layer in self.layers:
            x = layer(x)

        logits = self.fc_out(x)

        return logits


if __name__ == "__main__":
    # Test the window model
    print("Testing GPTWindowModel...")
    model = GPTWindowModel(128, 8, 4, 1000, 512, window_size=64, dilation=4)
    B = 8
    S = 256
    x = torch.randint(1000, (B, S))
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Test passed!")
