from typing import Optional

import jax.numpy as jnp
from flax import linen as nn


class gMLP(nn.Module):
    """
    MLP with spatial gating: https://arxiv.org/abs/2105.08050

    Optionally can enhance spatial gate with "tiny" attention component if
    `attn_features` is given. The output of this attention is added to gate weights
    in SGU before elementwise multiplication.
    """

    features: int
    attn_features: Optional[int] = None

    @nn.compact
    def __call__(self, x):
        shortcut = x
        x = nn.LayerNorm()(x)
        attn_out = None
        if self.attn_features:
            attn_out = TinyAttn(
                features=self.features // 2, attn_features=self.attn_features
            )(x)
        x = nn.Dense(self.features)(x)
        x = nn.gelu(x)
        x = SGU()(x, attn_out)
        x = nn.Dense(shortcut.shape[-1])(x)
        return x + shortcut


class SGU(nn.Module):
    """
    Spatial gated unit.

    Splits input along feature dimension to form features and gates a la GLU,
    but the gating is performed "spatially" (i.e. gate tokens, not features). The
    gate weight matrix rank is equal to input sequence length, and the model can not
    generalize to longer sequences without truncation. This depends heavily on
    spatial position inductive bias, and cannot work for i.e. graph data where
    outputs should be permutation invariant.

    The weights and biases of spatial projection are initialized to near 0 and 1,
    so that the gating op is identity function at start of training.
    """

    @nn.compact
    def __call__(self, x, additive_gate=None):
        x, gate = jnp.split(x, 2, axis=-1)
        gate = nn.LayerNorm()(gate)
        gate = nn.DenseGeneral(
            x.shape[-2],
            axis=-2,
            kernel_init=nn.initializers.normal(stddev=1e-4),
            bias_init=nn.initializers.ones,
            name="spatial_proj",
        )(gate)
        gate = gate.swapaxes(-2, -1)
        if additive_gate is not None:
            gate = gate + additive_gate
        return x * gate


class TinyAttn(nn.Module):
    """
    Tiny single headed attention for content aware (as opposed to purely spatial)
    gating.
    """

    features: int
    attn_features: int

    @nn.compact
    def __call__(self, x):
        qkv = nn.Dense(self.attn_features * 3)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        w = jnp.einsum("bnd,bmd->bnm", q, k)
        a = nn.softmax(w * self.attn_features ** -0.5)
        x = jnp.einsum("bnm,bmd->bnd", a, v)
        return nn.Dense(self.features)(x)


