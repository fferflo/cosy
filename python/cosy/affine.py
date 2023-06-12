import cosy, importlib, numpy, importlib.util
from pyquaternion import Quaternion

@cosy.with_np
def rotation_matrix_to_angle(rotation_matrix, np):
    return np.arctan2(rotation_matrix[..., 1, 0], rotation_matrix[..., 0, 0])

@cosy.with_np
def angle_to_rotation_matrix(angle, np):
    return np.stack([
        np.stack([np.cos(angle), -np.sin(angle)], axis=-1),
        np.stack([np.sin(angle), np.cos(angle)], axis=-1),
    ], axis=-2)

@cosy.with_np
def angle(v1, v2, np):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    # TODO: this doesnt work for arrays with more than one element
    if angle > np.pi:
        angle = angle - 2 * np.pi
    if angle <= -np.pi:
        angle = angle + 2 * np.pi

    return angle

@cosy.with_np
def angle_diff(a1, a2, np):
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    # TODO: this doesnt work for arrays with more than one element
    diff = np.abs(a1 - a2)
    while diff < 0.0:
        diff += 2 * np.pi
    while diff >= 2 * np.pi:
        diff -= 2 * np.pi

    if diff >= np.pi:
        diff = 2 * np.pi - diff

    return diff

@cosy.with_np
def rotation_matrix_between_vectors(v1, v2, np):
    v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)

    sin = np.cross(v1, v2)
    cos = np.sum(v1 * v2, axis=-1)
    return np.stack([
        np.stack([cos, -sin], axis=-1),
        np.stack([sin, cos], axis=-1),
    ], axis=-2)

def project_2d_to_3d(transform, axes=[0, 1]):
    np = transform.np
    if isinstance(transform, Rigid):
        rotation = np.eye(3, dtype=transform.dtype)
        for xy in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            rotation[axes[xy[0]], axes[xy[1]]] = transform.rotation[xy[0], xy[1]]
        translation = np.zeros((3,), dtype=transform.dtype)
        translation[axes[0]] = transform.translation[0]
        translation[axes[1]] = transform.translation[1]
        return Rigid(rotation=rotation, translation=translation)
    elif isinstance(transform, ScaledRigid):
        rotation = np.eye(3, dtype=transform.dtype)
        for xy in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            rotation[axes[xy[0]], axes[xy[1]]] = transform.rotation[xy[0], xy[1]]
        translation = np.zeros((3,), dtype=transform.dtype)
        translation[axes[0]] = transform.translation[0]
        translation[axes[1]] = transform.translation[1]
        return ScaledRigid(rotation=rotation, translation=translation, scale=transform.scale)
    else:
        raise ValueError("Invalid transform")

def stack(transforms, axis=0):
    if len(set(type(t) for t in transforms)) > 1:
        raise ValueError("Transforms must have the same type")
    np = transforms[0].np
    if isinstance(transforms[0], Rigid):
        return Rigid(
            rotation=np.stack([t.rotation for t in transforms], axis=axis),
            translation=np.stack([t.translation for t in transforms], axis=axis),
        )
    elif isinstance(transforms[0], ScaledRigid):
        return ScaledRigid(
            rotation=np.stack([t.rotation for t in transforms], axis=axis),
            translation=np.stack([t.translation for t in transforms], axis=axis),
            scale=np.stack([t.scale for t in transforms], axis=axis),
        )
    else:
        raise ValueError("Invalid transforms")

def concatenate(transforms, axis=0):
    if len(set(type(t) for t in transforms)) > 1:
        raise ValueError("Transforms must have the same type")
    np = transforms[0].np
    if isinstance(transforms[0], Rigid):
        return Rigid(
            rotation=np.concatenate([t.rotation for t in transforms], axis=axis),
            translation=np.concatenate([t.translation for t in transforms], axis=axis),
        )
    elif isinstance(transforms[0], ScaledRigid):
        return ScaledRigid(
            rotation=np.concatenate([t.rotation for t in transforms], axis=axis),
            translation=np.concatenate([t.translation for t in transforms], axis=axis),
            scale=np.concatenate([t.scale for t in transforms], axis=axis),
        )
    else:
        raise ValueError("Invalid transforms")

def pad(transform, pad_width, **kwargs):
    np = transform.np
    if isinstance(transform, Rigid):
        return Rigid(
            rotation=np.pad(transform.rotation, pad_width=list(pad_width) + [[0, 0], [0, 0]], **kwargs),
            translation=np.pad(transform.translation, pad_width=list(pad_width) + [[0, 0]], **kwargs),
        )
    elif isinstance(transform, ScaledRigid):
        return ScaledRigid(
            rotation=np.pad(transform.rotation, pad_width=list(pad_width) + [[0, 0], [0, 0]], **kwargs),
            translation=np.pad(transform.translation, pad_width=list(pad_width) + [[0, 0]], **kwargs),
            scale=np.pad(transform.scale, pad_width=list(pad_width), **kwargs),
        )
    else:
        raise ValueError("Invalid transform")

def slice(transform, slice):
    np = transform.np
    if isinstance(transform, Rigid):
        return Rigid(
            rotation=transform.rotation[slice],
            translation=transform.translation[slice],
        )
    elif isinstance(transform, ScaledRigid):
        return ScaledRigid(
            rotation=transform.rotation[slice],
            translation=transform.translation[slice],
            scale=transform.scale[slice],
        )
    else:
        raise ValueError("Invalid transform")

class Rigid:
    def __init__(self, rotation=None, translation=None, dtype=None, rank=None, batchshape=None):
        self.np = cosy.deduce_module(translation, rotation)
        if isinstance(rotation, float) or isinstance(rotation, int):
            rotation = angle_to_rotation_matrix(float(rotation))
        if not rotation is None:
            rotation = self.np.asarray(rotation)
        if not translation is None:
            translation = self.np.asarray(translation)

        if dtype is None:
            if not rotation is None:
                dtype = rotation.dtype
            elif not translation is None:
                dtype = translation.dtype
            else:
                raise ValueError("Expected dtype argument when rotation and translation are not given")
        if not isinstance(dtype, str):
            s = str(dtype)
            if s.isalnum():
                dtype = s
            else:
                s = dtype.__repr__().split(".")[-1] # Tensorflow dtypes
                if s.isalnum():
                    dtype = s
                else:
                    raise ValueError(f"Invalid dtype {dtype}")

        if rank is None:
            if not translation is None:
                rank = translation.shape[-1]
            elif not rotation is None:
                rank = rotation.shape[-1]
            else:
                raise ValueError("Expected rank argument when rotation and translation are not given")

        if batchshape is None:
            if not translation is None:
                batchshape = translation.shape[:-1]
            elif not rotation is None:
                batchshape = rotation.shape[:-2]
            else:
                batchshape = ()

        if translation is None:
            translation = self.np.zeros(batchshape + (rank,), dtype=dtype)
        if rotation is None:
            rotation = self.np.eye(rank, dtype=dtype)
            for _ in range(len(batchshape)):
                rotation = rotation[self.np.newaxis]
            rotation = self.np.broadcast_to(rotation, batchshape + (rank, rank))
        if rotation.shape != batchshape + (rank, rank):
            raise ValueError(f"Got invalid rotation shape {rotation.shape}")
        if translation.shape != batchshape + (rank,):
            raise ValueError(f"Got invalid translation shape {translation.shape}")

        if self.np == numpy:
            if not self.np.all(self.np.isfinite(rotation)):
                raise ValueError(f"Got non-finite rotation {rotation}")
            if not self.np.all(self.np.isfinite(translation)):
                raise ValueError(f"Got non-finite translation {translation}")
            if not self.np.allclose(self.np.matmul(rotation, self._transpose(rotation, batchshape=batchshape)), self.np.eye(rank), atol=1e-6):
                error = self.np.sum(self.np.matmul(rotation, self._transpose(rotation, batchshape=batchshape)) - self.np.eye(rank))
                raise ValueError(f"Got non-orthogonal rotation {rotation} with error={error} and dtype {dtype}")

        self.rotation = rotation.astype(dtype)
        self.translation = translation.astype(dtype)

    batchshape = property(lambda self: self.translation.shape[:-1])
    rank = property(lambda self: self.translation.shape[-1])
    dtype = property(lambda self: self.translation.dtype)

    def _transpose(self, x, batchshape=None):
        if batchshape is None:
            batchshape = self.batchshape
        n = len(batchshape)
        permutation = tuple(list(range(n))) + (n + 1, n)
        return self.np.transpose(x, permutation)

    def __call__(self, points):
        if isinstance(points, tuple) or isinstance(points, list):
            points = self.np.asarray(points)
        points = points.astype(self.dtype)

        has_points_axis = True
        if len(points.shape) == 1:
            has_points_axis = False
            points = points[self.np.newaxis, :]
        if len(points.shape) == 2:
            for _ in range(len(self.batchshape)):
                points = points[self.np.newaxis]
            points = self.np.broadcast_to(points, self.batchshape + points.shape[-2:])
        if points.shape == self.batchshape + (self.rank,):
            has_points_axis = False
            points = points[..., self.np.newaxis, :]
        if len(points.shape) != len(self.batchshape) + 2 or points.shape[:-2] != self.batchshape or points.shape[-1] != self.rank:
            raise ValueError(f"Invalid points shape {points.shape}")

        points = self._transpose(self.np.matmul(self.rotation, self._transpose(points)))
        points = points + self.translation[..., self.np.newaxis, :]

        if not has_points_axis:
            points = points[..., 0, :]

        return points

    def __str__(self):
        return f"{{t={self.translation.tolist()} R={self.rotation.tolist()}}}"

    def __mul__(self, other):
        if isinstance(other, ScaledRigid):
            return ScaledRigid(self) * other
        else:
            return Rigid(self.np.matmul(self.rotation, other.rotation), self(other.translation), dtype=self.dtype)

    def inverse(self):
        inv_rotation = self._transpose(self.rotation)
        inv_translation = self.np.matmul(inv_rotation, -self.translation[..., self.np.newaxis])[..., 0]
        return Rigid(inv_rotation, inv_translation, dtype=self.dtype)

    def __truediv__(self, other):
        return self * other.inverse()

    def astype(self, dtype):
        return Rigid(self.rotation, self.translation, dtype=dtype)

    @staticmethod
    @cosy.with_np
    def least_squares(from_points, to_points, np):
        from_points = np.asarray(from_points)
        to_points = np.asarray(to_points)
        assert from_points.shape == to_points.shape
        rank = from_points.shape[1]

        mean1 = np.mean(from_points, axis=0)
        mean2 = np.mean(to_points, axis=0)
        from_points = from_points - mean1
        to_points = to_points - mean2

        W = np.matmul(np.transpose(to_points, (1, 0)), from_points)

        if rank == 2:
            y = 0.5 * (W[1, 0] - W[0, 1])
            x = 0.5 * (W[0, 0] + W[1, 1])
            r = np.sqrt(x * x + y * y)
            rotation_matrix = np.stack([
                np.stack([x, -y]),
                np.stack([y, x])
            ]) * (1.0 / r)
        else:
            u, s, vT = np.linalg.svd(W)
            rotation_matrix = np.dot(vT.T, u.T)

        translation = mean2 - np.matmul(rotation_matrix, mean1)

        return Rigid(rotation_matrix, translation)

    @staticmethod
    @cosy.with_np
    def from_matrix(m, np):
        rank = m.shape[-1] - 1
        if not np.allclose(m[..., rank, :-1], 0.0) or not np.allclose(m[..., rank, rank], 1.0):
            raise ValueError("Not a valid Rigid transformation matrix")
        return Rigid(
            rotation=m[..., :rank, :rank],
            translation=m[..., :rank, rank]
        )

    def to_matrix(self):
        rank = self.rank
        m = self.np.concatenate([self.rotation, self.translation[..., :, self.np.newaxis]], axis=-1)
        r = self.np.asarray([0] * rank + [1])
        for _ in range(len(self.batchshape)):
            r = r[self.np.newaxis]
        r = self.np.broadcast_to(r, self.batchshape + (rank + 1,))
        m = self.np.concatenate([m, r[..., self.np.newaxis, :]], axis=-2)
        return m

    @staticmethod
    def slerp(transform1, transform2=None, amount=0.5):
        assert 0 <= amount and amount <= 1
        if transform2 is None:
            transform2 = transform1
            transform1 = Rigid(dtype=transform1.dtype, rank=transform1.rank) # Identity
        assert transform1.rank == transform2.rank
        assert transform1.np == numpy and transform2.np == numpy # TODO: not implemented yet for other nps

        if amount == 0:
            return transform1
        elif amount == 1:
            return transform2

        if transform1.rank() == 2:
            angle1 = rotation_matrix_to_angle(transform1.rotation)
            angle2 = rotation_matrix_to_angle(transform2.rotation)
            return Rigid(
                rotation=angle1 + amount * (angle2 - angle1),
                translation=transform1.translation + amount * (transform2.translation - transform1.translation),
            )
        elif transform1.rank() == 3:
            q1 = Quaternion(matrix=transform1.rotation, rtol=1e-04, atol=1e-06)
            q2 = Quaternion(matrix=transform2.rotation, rtol=1e-04, atol=1e-06)
            q = Quaternion.slerp(q1, q2, amount)
            return Rigid(
                rotation=q.rotation_matrix,
                translation=transform1.translation + amount * (transform2.translation - transform1.translation),
            )
        else:
            raise ValueError(f"Slerp for transformation with rank {self.rank} not supported")

if not importlib.util.find_spec("jax") is None:
    def flatten(transform):
        return [transform.rotation, transform.translation], []

    def unflatten(aux_data, children):
        transform = Rigid.__new__(Rigid)
        transform.rotation = children[0]
        transform.translation = children[1]

        if not (type(transform.rotation) is object or transform.rotation is None or isinstance(transform.rotation, Rigid)):
            transform.np = cosy.deduce_module(transform.translation, transform.rotation)
        else:
            transform.np = None

        return transform

    import jax.tree_util
    jax.tree_util.register_pytree_node(Rigid, flatten, unflatten)













class ScaledRigid: # TODO: implement for nd scale
    def __init__(self, rotation=None, translation=None, scale=None, dtype=None, rank=None, batchshape=None):
        self.np = cosy.deduce_module(translation, rotation)
        if isinstance(rotation, Rigid) and translation is None and scale is None and dtype is None and batchshape is None:
            rigid = rotation
            rotation = rigid.rotation
            translation = rigid.translation
            dtype = rigid.dtype
            scale = 1

        if isinstance(rotation, float) or isinstance(rotation, int):
            rotation = angle_to_rotation_matrix(float(rotation))
        if not rotation is None:
            rotation = self.np.asarray(rotation)
        if not translation is None:
            translation = self.np.asarray(translation)
        if not scale is None:
            scale = self.np.asarray(scale)

        if dtype is None:
            if not rotation is None:
                dtype = rotation.dtype
            elif not translation is None:
                dtype = translation.dtype
            elif not scale is None:
                dtype = scale.dtype
            else:
                raise ValueError("Expected dtype argument when rotation, translation and scale are not given")
        if not isinstance(dtype, str):
            s = str(dtype)
            if s.isalnum():
                dtype = s
            else:
                s = dtype.__repr__().split(".")[-1] # Tensorflow dtypes
                if s.isalnum():
                    dtype = s
                else:
                    raise ValueError(f"Invalid dtype {dtype}")

        if rank is None:
            if not translation is None:
                rank = translation.shape[-1]
            elif not rotation is None:
                rank = rotation.shape[-1]
            else:
                raise ValueError("Expected rank argument when rotation and translation are not given")

        if batchshape is None:
            if not translation is None:
                batchshape = translation.shape[:-1]
            elif not rotation is None:
                batchshape = rotation.shape[:-2]
            else:
                batchshape = ()

        if translation is None:
            translation = self.np.zeros(batchshape + (rank,), dtype=dtype)
        if rotation is None:
            rotation = self.np.eye(rank, dtype=dtype)
            for _ in range(len(batchshape)):
                rotation = rotation[self.np.newaxis]
            rotation = self.np.broadcast_to(rotation, batchshape + (rank, rank))
        if scale is None:
            scale = self.np.ones(batchshape, dtype=dtype)
        if rotation.shape != batchshape + (rank, rank):
            raise ValueError(f"Got invalid rotation shape {rotation.shape}")
        if translation.shape != batchshape + (rank,):
            raise ValueError(f"Got invalid translation shape {translation.shape}")

        if self.np == numpy:
            if not self.np.all(self.np.isfinite(rotation)):
                raise ValueError(f"Got non-finite rotation {rotation}")
            if not self.np.all(self.np.isfinite(translation)):
                raise ValueError(f"Got non-finite translation {translation}")
            if not self.np.all(self.np.isfinite(scale)):
                raise ValueError(f"Got non-finite scale {scale}")
            if not self.np.allclose(self.np.matmul(rotation, self._transpose(rotation, batchshape=batchshape)), self.np.eye(rank), atol=1e-6):
                error = self.np.sum(self.np.matmul(rotation, self._transpose(rotation, batchshape=batchshape)) - self.np.eye(rank))
                raise ValueError(f"Got non-orthogonal rotation {rotation} with error={error} and dtype {dtype}")

        self.rotation = rotation.astype(dtype)
        self.translation = translation.astype(dtype)
        self.scale = scale.astype(dtype)

    batchshape = property(lambda self: self.translation.shape[:-1])
    rank = property(lambda self: self.translation.shape[-1])
    dtype = property(lambda self: self.translation.dtype)

    def _transpose(self, x, batchshape=None):
        if batchshape is None:
            batchshape = self.batchshape
        n = len(batchshape)
        permutation = tuple(list(range(n))) + (n + 1, n)
        return self.np.transpose(x, permutation)

    def __call__(self, points):
        if isinstance(points, tuple) or isinstance(points, list):
            points = self.np.asarray(points)
        points = points.astype(self.dtype)

        has_points_axis = True
        if len(points.shape) == 1:
            has_points_axis = False
            points = points[self.np.newaxis, :]
        if len(points.shape) == 2:
            for _ in range(len(self.batchshape)):
                points = points[self.np.newaxis]
            points = self.np.broadcast_to(points, self.batchshape + points.shape[-2:])
        if points.shape == self.batchshape + (self.rank,):
            has_points_axis = False
            points = points[..., self.np.newaxis, :]
        if len(points.shape) != len(self.batchshape) + 2 or points.shape[:-2] != self.batchshape or points.shape[-1] != self.rank:
            raise ValueError(f"Invalid points shape {points.shape}")

        points = self._transpose(self.np.matmul(self.scale[..., self.np.newaxis, self.np.newaxis] * self.rotation, self._transpose(points)))
        points = points + self.translation[..., self.np.newaxis, :]

        if not has_points_axis:
            points = points[..., 0, :]

        return points

    def __str__(self):
        return f"{{t={self.translation.tolist()} R={self.rotation.tolist()} s={self.scale.tolist()}}}"

    def __mul__(self, other):
        if isinstance(other, Rigid):
            other = ScaledRigid(other)
        return ScaledRigid(self.np.matmul(self.rotation, other.rotation), self(other.translation), self.scale * other.scale, dtype=self.dtype)

    def inverse(self):
        inv_rotation = self._transpose(self.rotation)
        inv_scale = 1.0 / self.scale
        inv_translation = inv_scale * self.np.matmul(inv_rotation, -self.translation[..., self.np.newaxis])[..., 0]
        return ScaledRigid(inv_rotation, inv_translation, inv_scale, dtype=self.dtype)

    def __truediv__(self, other):
        return self * other.inverse()

    def astype(self, dtype):
        return ScaledRigid(self.rotation, self.translation, self.scale, dtype=dtype)

    @staticmethod
    @cosy.with_np
    def least_squares(from_points, to_points, np):
        from_points = np.asarray(from_points)
        to_points = np.asarray(to_points)

        mean1 = np.mean(from_points, axis=0)
        mean2 = np.mean(to_points, axis=0)
        from_points = from_points - mean1
        to_points = to_points - mean2

        W = np.matmul(np.transpose(to_points, (1, 0)), from_points)

        y = 0.5 * (W[1, 0] - W[0, 1])
        x = 0.5 * (W[0, 0] + W[1, 1])
        r = np.sqrt(x * x + y * y)
        rotation_matrix = np.stack([
            np.stack([x, -y]),
            np.stack([y, x])
        ]) * (1.0 / r)

        from_points = np.transpose(np.matmul(rotation_matrix, np.transpose(from_points, (1, 0))), (1, 0))

        scale = np.sum(from_points * to_points) / np.sum(from_points * from_points)
        assert scale > 0

        translation = mean2 - scale * np.matmul(rotation_matrix, mean1)

        return ScaledRigid(rotation_matrix, translation, scale)

    def to_matrix(self):
        rank = self.rank
        m = self.np.concatenate([self.scale[..., self.np.newaxis, self.np.newaxis] * self.rotation, self.translation[..., :, self.np.newaxis]], axis=-1)
        r = self.np.asarray([0] * rank + [1])
        for _ in range(len(self.batchshape)):
            r = r[self.np.newaxis]
        r = self.np.broadcast_to(r, self.batchshape + (rank + 1,))
        m = self.np.concatenate([m, r[..., self.np.newaxis, :]], axis=-2)
        return m

if not importlib.util.find_spec("jax") is None:
    def flatten(transform):
        return [transform.rotation, transform.translation, transform.scale], []

    def unflatten(aux_data, children):
        transform = ScaledRigid.__new__(ScaledRigid)
        transform.rotation = children[0]
        transform.translation = children[1]
        transform.scale = children[2]
        if not (type(transform.rotation) is object or transform.rotation is None or isinstance(transform.rotation, ScaledRigid)):
            transform.np = cosy.deduce_module(transform.translation, transform.rotation, transform.scale)
        else:
            transform.np = None
        return transform

    import jax.tree_util
    jax.tree_util.register_pytree_node(ScaledRigid, flatten, unflatten)
