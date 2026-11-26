"""NoiseWrapper: geometry-only jitter and whole-instance missed detections."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxtari
from jaxtari.environment import ObjectObservation
from jaxtari.wrappers import AtariWrapper, NoiseWrapper

GAMES = ("kangaroo", "pong", "seaquest", "breakout", "freeway", "frostbite")
JITTERED = NoiseWrapper.JITTERED_FIELDS


def _objects(obs):
    is_object = lambda node: isinstance(node, ObjectObservation)
    return [n for n in jax.tree.leaves(obs, is_leaf=is_object) if is_object(n)]


def _noisy(game, detect_prob, std_dev, seed=0):
    env = AtariWrapper(jaxtari.make(game), noop_max=0, first_fire=False)
    clean, state = env.reset(jax.random.PRNGKey(seed))
    wrapper = NoiseWrapper(env, detect_prob=detect_prob, std_dev=std_dev)
    return clean, wrapper._add_noise_to_obs(clean, jax.random.PRNGKey(seed + 1))


@pytest.mark.parametrize("game", GAMES)
def test_structure_and_shapes_are_preserved(game):
    clean, noisy = _noisy(game, 0.5, 1.0)
    assert jax.tree.structure(clean) == jax.tree.structure(noisy)
    for a, b in zip(jax.tree.leaves(clean), jax.tree.leaves(noisy)):
        assert jnp.shape(a) == jnp.shape(b)
        assert b.dtype == jnp.float32


def test_no_noise_is_identity():
    clean, noisy = _noisy("kangaroo", 1.0, 0.0)
    for a, b in zip(jax.tree.leaves(clean), jax.tree.leaves(noisy)):
        np.testing.assert_array_equal(np.asarray(a, np.float32), b)


def test_jitter_touches_geometry_only_and_keeps_sizes_non_negative():
    clean, noisy = _noisy("kangaroo", 1.0, 2.0)
    moved = False
    for c, n in zip(_objects(clean), _objects(noisy)):
        for name in c.__dataclass_fields__:
            a = np.asarray(getattr(c, name), np.float32)
            b = np.asarray(getattr(n, name))
            if name in JITTERED:
                moved |= bool(np.any(a != b))
            else:
                np.testing.assert_array_equal(a, b, err_msg=name)
        assert np.all(np.asarray(n.width) >= 0) and np.all(np.asarray(n.height) >= 0)
    assert moved


def test_missed_detection_zeroes_whole_instances():
    clean, noisy = _noisy("kangaroo", 0.5, 0.0)
    dropped = kept = 0
    for c, n in zip(_objects(clean), _objects(noisy)):
        fields = list(c.__dataclass_fields__)
        a = np.stack([np.broadcast_to(np.asarray(getattr(c, f), np.float32), np.shape(c.x))
                      for f in fields], -1)
        b = np.stack([np.broadcast_to(np.asarray(getattr(n, f)), np.shape(c.x))
                      for f in fields], -1)
        a, b = a.reshape(-1, len(fields)), b.reshape(-1, len(fields))
        for clean_row, noisy_row in zip(a, b):
            if np.array_equal(clean_row, noisy_row):
                kept += 1
            else:
                # a miss never removes single fields: the whole instance reads 0
                np.testing.assert_array_equal(noisy_row, 0.0)
                dropped += 1
    assert dropped and kept


def test_detect_prob_zero_drops_everything():
    _, noisy = _noisy("kangaroo", 0.0, 1.0)
    for obj in _objects(noisy):
        for leaf in jax.tree.leaves(obj):
            np.testing.assert_array_equal(leaf, 0.0)
