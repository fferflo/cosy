import cosy, math, importlib

def _test(np):
    assert np.allclose(cosy.angle(np.asarray([3.0, 0.0]), [0.0, 1.0]), math.pi / 2)
    assert np.allclose(cosy.angle(np.asarray([1.0, 0.0]), [0.0, -5.0]), -math.pi / 2)

    t1 = cosy.Rigid(rotation=cosy.angle_to_rotation_matrix(2.0, np=np), translation=[1.0, 2.0])
    t_id = cosy.Rigid(dtype=t1.dtype, rank=2)

    assert np.allclose(t_id.to_matrix(), (t1 * t1.inverse()).to_matrix(), atol=1e-06)

    t1 = cosy.stack([t1, t1])
    t1 = cosy.stack([t1, t1])
    assert t1.batchshape == (2, 2)

    assert np.allclose(t1.to_matrix(), cosy.Rigid.from_matrix(t1.to_matrix()).to_matrix())

    v1 = np.asarray([1.0, 0.0])
    v2 = np.asarray([10.0, 3.0])
    v2 = v2 / np.linalg.norm(v2, keepdims=True)

    assert np.allclose(cosy.rotation_matrix_between_vectors(v1, v2) @ v1, v2, atol=1e-5)
    assert np.allclose(cosy.angle_to_rotation_matrix(cosy.angle(v1, v2)) @ v1, v2, atol=1e-5)

    assert np.allclose(cosy.rotation_matrix_between_vectors(v2, v1) @ v2, v1, atol=1e-5)
    assert np.allclose(cosy.angle_to_rotation_matrix(cosy.angle(v2, v1)) @ v2, v1, atol=1e-5)

import numpy as np
def test_np():
    _test(np)

# Tensorflow numpy interface currently is incomplete
# if not importlib.util.find_spec("tensorflow") is None:
#     import tensorflow.experimental.numpy as tnp
#     from tensorflow.python.ops.numpy_ops import np_config
#     np_config.enable_numpy_behavior()
#     def test_tf():
#         _test(tnp)

if not importlib.util.find_spec("jax") is None:
    import jax.numpy as jnp
    import jax
    def test_jax():
        _test(jnp)

        transform = cosy.Rigid(rotation=cosy.angle_to_rotation_matrix(2.0), translation=[1.0, 2.0])
        def func(transform):
            return transform
        transform = jax.jit(func)(transform)
