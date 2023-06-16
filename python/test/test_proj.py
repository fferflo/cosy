import cosy, math, pickle
import numpy as np

def test_transformer():
    epsg3857 = cosy.proj.CRS("epsg:3857")
    epsg4326 = cosy.proj.CRS("epsg:4326")
    epsg4326_to_epsg3857 = cosy.proj.Transformer(epsg4326, epsg3857)

    latlon = np.asarray([49.011384, 8.416419])

    epsg4326_to_epsg3857.transform(latlon.astype("float32"))
    epsg4326_to_epsg3857.transform(latlon.astype("float64"))
    epsg4326_to_epsg3857.transform((latlon[0], latlon[1]))
    epsg4326_to_epsg3857.transform((float(latlon[0]), float(latlon[1])))
    epsg4326_to_epsg3857.transform([latlon[0], latlon[1]])
    epsg4326_to_epsg3857.transform([float(latlon[0]), float(latlon[1])])

    assert np.allclose(latlon, epsg4326_to_epsg3857.transform_inverse(epsg4326_to_epsg3857.transform(latlon)))
    assert np.allclose(latlon, epsg4326_to_epsg3857.inverse()(epsg4326_to_epsg3857(latlon)))

    assert np.allclose(epsg3857.get_vector("north"), np.asarray([0.0, 1.0]))
    assert np.allclose(epsg4326.get_vector("north"), np.asarray([1.0, 0.0]))

    def bearing_to_vector(bearing):
        return np.dot(cosy.angle_to_rotation_matrix(epsg4326_to_epsg3857.transform_angle(bearing)), np.asarray([1.0, 0.0]))
    assert np.allclose(bearing_to_vector(0.0),         epsg3857.get_vector("north"))
    assert np.allclose(bearing_to_vector(math.pi / 2), epsg3857.get_vector("east"))

    def vector_to_bearing(vector):
        return cosy.angle_between_vectors(epsg4326.get_vector("north"), epsg4326_to_epsg3857.transform_inverse(vector))
    assert np.allclose(vector_to_bearing(epsg3857.get_vector("north")), 0)
    assert np.allclose(vector_to_bearing(epsg3857.get_vector("east")), math.pi / 2)

    assert np.allclose(epsg4326_to_epsg3857.transform_angle(0.0),         cosy.angle_between_vectors(np.asarray([1.0, 0.0]), epsg3857.get_vector("north")))
    assert np.allclose(epsg4326_to_epsg3857.transform_angle(math.pi / 2), cosy.angle_between_vectors(np.asarray([1.0, 0.0]), epsg3857.get_vector("east")))

    assert np.allclose(epsg4326_to_epsg3857.transform_angle(epsg4326_to_epsg3857.transform_angle_inverse(0.0)), 0.0)
    assert np.allclose(epsg4326_to_epsg3857.transform_angle(epsg4326_to_epsg3857.transform_angle_inverse(math.pi / 2)), math.pi / 2)

def test_pickle():
    epsg3857 = cosy.proj.CRS("epsg:3857")
    epsg4326 = cosy.proj.CRS("epsg:4326")
    epsg4326_to_epsg3857 = cosy.proj.Transformer(epsg4326, epsg3857)

    pickle.loads(pickle.dumps(epsg3857))
    pickle.loads(pickle.dumps(epsg4326_to_epsg3857))
