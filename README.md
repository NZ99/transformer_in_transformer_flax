## Transformer in Transformer in JAX/Flax

This repository implements <a href="https://arxiv.org/abs/2103.00112">Transformer in Transformer</a>, pixel level attention paired with patch level attention for image classification. It is heavily inspired by both <a href="https://github.com/lucidrains/transformer-in-transformer">lucidrains's</a> Pytorch implementation and <a href="https://github.com/google-research/vision_transformer">Google Brain's</a> Vision Transformer repo.

<a href="https://www.youtube.com/watch?v=HWna2c5VXDg">AI Coffee Break with Letita</a>

## Install

```bash
$ pip install transformer-in-transformer-flax
```

## Usage

```python
from jax import random
from jax import numpy as jnp
from transformer_in_transformer_flax import TransformerInTransformer, TNTConfig

#example configuration for TNT-B
config = TNTConfig(
    num_classes = 1000,
    depth = 12,
    image_size = 224,
    patch_size = 16,
    transformed_patch_size = 4,
    inner_dim = 40,
    inner_heads = 4,
    inner_dim_head = 64,
    inner_r = 4,
    outer_dim = 640,
    outer_heads = 10,
    outer_dim_head = 64,
    outer_r = 4
)

rng = random.PRNGKey(seed=0)
model = TransformerInTransformer(config=config)
params = model.init(rng, jnp.ones((1, 224, 224, 3), dtype=config.dtype))
img = random.uniform(rng, (2, 224, 224, 3))
logits = model.apply(params, img) # (2, 1000)
```

## Citations

```bibtex
@misc{han2021transformer,
    title   = {Transformer in Transformer}, 
    author  = {Kai Han and An Xiao and Enhua Wu and Jianyuan Guo and Chunjing Xu and Yunhe Wang},
    year    = {2021},
    eprint  = {2103.00112},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```