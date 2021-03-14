from typing import Callable, Any

from jax import numpy as jnp
from jax.nn import initializers

from flax import linen as nn
from flax import struct

from einops import rearrange


@struct.dataclass
class TNTConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
    num_classes: int = 1000
    depth: int = 12
    image_size: int = 224
    patch_size: int = 16
    transformed_patch_size: int = 4
    inner_dim: int = 40
    inner_heads: int = 4
    inner_dim_head: int = 64
    inner_r: int = 4
    outer_dim: int = 640
    outer_heads: int = 10
    outer_dim_head: int = 64
    outer_r: int = 4
    dtype: Any = jnp.float32
    kernel_init: Callable = initializers.xavier_uniform()
    bias_init: Callable = initializers.normal(stddev=1e-6)
    posemb_init: Callable = initializers.normal(stddev=0.02)


class AddPositionEmbs(nn.Module):

    config: TNTConfig
    patch: bool

    @nn.compact
    def __call__(self, inputs):
        cfg = self.config
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                  ' but it is: %d' % inputs.ndim)
        if self.patch:
            pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
            pe = self.param('patch_pos_embedding', cfg.posemb_init,
                            pos_emb_shape)
            return inputs + pe
        else:
            pos_emb_shape = (1, cfg.transformed_patch_size**2, inputs.shape[-1])
            pe = self.param('pixel_pos_embedding', cfg.posemb_init,
                            pos_emb_shape)
            return inputs + pe


class MlpBlock(nn.Module):

    config: TNTConfig
    inner: bool

    @nn.compact
    def __call__(self, inputs):

        cfg = self.config

        if self.inner:
            dim = cfg.inner_dim
            r = cfg.inner_r
        else:
            dim = cfg.outer_dim
            r = cfg.outer_r

        x = nn.Dense(dim * r,
                     dtype=cfg.dtype,
                     kernel_init=cfg.kernel_init,
                     bias_init=cfg.bias_init)(inputs)
        x = nn.gelu(x)
        output = nn.Dense(dim,
                          dtype=cfg.dtype,
                          kernel_init=cfg.kernel_init,
                          bias_init=cfg.bias_init)(x)
        return output


class TransformPatches(nn.Module):

    config: TNTConfig

    @nn.compact
    def __call__(self, inputs):

        cfg = self.config
        x = rearrange(inputs,
                      'b (h p1) (w p2) c -> (b h w) p1 p2 c',
                      p1=cfg.patch_size,
                      p2=cfg.patch_size)
        x = rearrange(x,
                      'b (h h2) (w w2) c -> b (h w) (c h2 w2)',
                      h2=cfg.transformed_patch_size,
                      w2=cfg.transformed_patch_size)
        output = nn.Dense(cfg.inner_dim,
                          dtype=cfg.dtype,
                          kernel_init=cfg.kernel_init,
                          bias_init=cfg.bias_init)(x)
        return output


class TNTBlock(nn.Module):

    config: TNTConfig

    @nn.compact
    def __call__(self, pixel_embeddings, patch_embeddings):

        cfg = self.config

        v = cfg.image_size // cfg.patch_size

        #Inner T-Block
        x = nn.LayerNorm(dtype=cfg.dtype)(pixel_embeddings)
        x = nn.SelfAttention(num_heads=cfg.inner_heads,
                             qkv_features=cfg.inner_heads * cfg.inner_dim_head,
                             out_features=cfg.inner_dim,
                             use_bias=False,
                             kernel_init=cfg.kernel_init,
                             deterministic=True)(x)
        x = x + pixel_embeddings
        y = nn.LayerNorm(dtype=cfg.dtype)(x)
        y = MlpBlock(config=cfg, inner=True)(y)
        inner_output = x + y

        x = rearrange(pixel_embeddings, '... n d -> ... (n d)')
        x = nn.Dense(cfg.outer_dim,
                     dtype=cfg.dtype,
                     kernel_init=cfg.kernel_init,
                     bias_init=cfg.bias_init)(x)
        x = rearrange(x, '(b h w) d -> b (h w) d', h=v, w=v)
        x = jnp.pad(x, ((0, 0), (0, 1), (0, 0)))
        x = x + patch_embeddings

        #Outer T-Block
        x = nn.LayerNorm(dtype=cfg.dtype)(x)
        x = nn.SelfAttention(num_heads=cfg.outer_heads,
                             qkv_features=cfg.outer_heads * cfg.outer_dim_head,
                             out_features=cfg.outer_dim,
                             use_bias=False,
                             kernel_init=cfg.kernel_init,
                             deterministic=True)(x)
        x = x + patch_embeddings
        y = nn.LayerNorm(dtype=cfg.dtype)(x)
        y = MlpBlock(config=cfg, inner=False)(y)
        outer_output = x + y

        return inner_output, outer_output


class Encoder(nn.Module):

    config: TNTConfig

    @nn.compact
    def __call__(self, pixel_embeddings, patch_embeddings):
        cfg = self.config

        assert pixel_embeddings.ndim == 3
        assert patch_embeddings.ndim == 3

        pixel_embeddings = AddPositionEmbs(config=cfg,
                                           patch=False)(pixel_embeddings)

        patch_embeddings = AddPositionEmbs(config=cfg,
                                           patch=True)(patch_embeddings)

        for layer in range(cfg.depth):
            pixel_embeddings, patch_embeddings = TNTBlock(config=cfg)(
                pixel_embeddings, patch_embeddings)

        return patch_embeddings


class TransformerInTransformer(nn.Module):

    config: TNTConfig

    @nn.compact
    def __call__(self, inputs):
        cfg = self.config

        assert cfg.image_size % cfg.patch_size == 0
        assert cfg.patch_size % cfg.transformed_patch_size == 0

        n, h, w, c = inputs.shape

        patch_embeddings = nn.Conv(cfg.outer_dim,
                                   kernel_size=(cfg.patch_size, cfg.patch_size),
                                   strides=(cfg.patch_size, cfg.patch_size),
                                   padding='VALID')(inputs)

        pixel_embeddings = TransformPatches(config=cfg)(inputs)

        n, h, w, c = patch_embeddings.shape
        patch_embeddings = rearrange(patch_embeddings, 'n h w c -> n (h w) c')

        cls = self.param('cls', initializers.zeros, (1, 1, c))
        cls = jnp.tile(cls, [n, 1, 1])

        patch_embeddings = jnp.concatenate([cls, patch_embeddings], axis=1)

        x = Encoder(config=cfg)(pixel_embeddings, patch_embeddings)
        x = x[:, 0]

        x = nn.Dense(cfg.num_classes,
                     dtype=cfg.dtype,
                     kernel_init=initializers.zeros,
                     bias_init=cfg.bias_init)(x)
        return x