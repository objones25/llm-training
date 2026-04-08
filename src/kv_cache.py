"""Key-value cache for autoregressive generation.

KVCache stores accumulated key and value tensors for each transformer layer,
enabling O(T) per-step generation cost instead of O(T²) by reusing prior
attention computations.

Two-phase generation
--------------------
1. **Prefill** — pass the full seed sequence through the model with an empty
   KVCache.  Each attention layer appends its K/V tensors into the cache,
   building a complete representation of the prompt.
2. **Generate** — feed one token at a time.  Each layer projects only the new
   token's Q/K/V, concatenates K/V onto the cache, and runs attention over
   the full cached history.  Cost per step is O(T_cached) instead of O(T²).

The training forward pass (``kv_cache=None``) is completely unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerKVCache:
    """Accumulated K and V tensors for a single transformer layer.

    Attributes
    ----------
    k : FloatTensor[B, n_heads, T_cached, head_dim]
    v : FloatTensor[B, n_heads, T_cached, head_dim]
    """

    k: torch.Tensor
    v: torch.Tensor

    @property
    def seq_len(self) -> int:
        """Number of tokens currently stored in this layer's cache."""
        return self.k.shape[2]


@dataclass
class KVCache:
    """Full KV cache across all transformer layers.

    Attributes
    ----------
    layers : list[LayerKVCache]
        One entry per transformer layer, in order.
    """

    layers: list[LayerKVCache]

    @classmethod
    def empty(
        cls,
        n_layers: int,
        batch_size: int,
        n_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "KVCache":
        """Create a KVCache with zero-length K/V tensors at each layer.

        Parameters
        ----------
        n_layers : int
            Number of transformer layers.
        batch_size : int
            Batch size B.  For generation this is typically 1.
        n_heads : int
            Number of attention heads.
        head_dim : int
            Dimension of each head (``d_model // n_heads``).
        device : torch.device
            Target device.
        dtype : torch.dtype, optional
            Floating-point dtype (default: float32).

        Returns
        -------
        KVCache
            Cache with shape ``[B, n_heads, 0, head_dim]`` at every layer.
        """
        layers = [
            LayerKVCache(
                k=torch.zeros(
                    batch_size, n_heads, 0, head_dim, device=device, dtype=dtype
                ),
                v=torch.zeros(
                    batch_size, n_heads, 0, head_dim, device=device, dtype=dtype
                ),
            )
            for _ in range(n_layers)
        ]
        return cls(layers=layers)

    @property
    def seq_len(self) -> int:
        """Number of tokens cached (all layers share the same length)."""
        return self.layers[0].seq_len
