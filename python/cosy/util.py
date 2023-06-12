import numpy as np
import sys, jax.tree_util

def deduce_module(*args, **kwargs):
    leaves, _ = jax.tree_util.tree_flatten((args, kwargs))
    any_jax = False
    if "jax" in sys.modules:
        import jax.numpy as jnp
        any_jax = any(isinstance(x, jnp.ndarray) for x in leaves)
    any_tf = False
    if "tensorflow" in sys.modules:
        import tensorflow.experimental.numpy as tnp
        import tensorflow as tf
        any_tf = any(tf.is_tensor(x) for x in leaves)

    if any_jax and any_tf:
        raise ValueError("Got arguments from multiple frameworks")
    elif any_jax:
        return jnp
    elif any_tf:
        return tnp
    else:
        return np

def with_np(func):
    def wrapper(*args, np=None, **kwargs):
        if np is None:
            np = deduce_module(args, kwargs)
        return func(*args, np=np, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper
