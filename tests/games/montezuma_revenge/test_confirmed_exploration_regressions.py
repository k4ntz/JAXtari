"""Independent native frames for fatal falls, ladder dispatch and support edges.

ALE expectations were captured before the fixes. These frozen mechanism states
do not assert regular reachability; fresh reset/room-start replay is separate.
"""
import json
from pathlib import Path

import pytest
import jax
import jax.numpy as jnp

from test_native_ale_regressions import _measured, _state, native_reset_state, native_step


REFERENCE = json.loads(Path(__file__).with_name('confirmed_exploration_ale_oracles.json').read_text())
FIELDS = REFERENCE['provenance']['fields']


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda c: c['name'])
def test_confirmed_native_event(native_reset_state, native_step, case):
    state = _state(native_reset_state, case['state'])
    assert len(case['actions']) == len(case['expected']) == len(case['rewards'])
    for tick, (action, expected, expected_reward) in enumerate(zip(
            case['actions'], case['expected'], case['rewards'])):
        _, state, reward, done, _ = native_step(state, jnp.int32(action))
        measured = _measured(state, done)
        assert [int(measured[k]) for k in FIELDS] == expected, (case['name'], tick)
        assert float(reward) == expected_reward, (case['name'], tick)
        # D8 is reused by other native handlers (including death animation).
        if int(state.native_mode) in (4, 7):
            assert int(state.native_fall_ticks) == case['fall_ticks'][tick], (case['name'], tick, 'D8')


def test_ladder_anchor_actions_under_jit_vmap(montezuma_env, native_reset_state):
    """The new dispatch state must batch with three distinct legal inputs."""
    cases = [next(c for c in REFERENCE['cases'] if c['name'] == 'ladder-down19-' + name)
             for name in ['fire', 'upfire', 'rightfire']]
    state = _state(native_reset_state, cases[0]['state'])
    _, state, _, _, _ = montezuma_env.step(state, jnp.int32(5))
    states = jax.tree.map(lambda v: jnp.broadcast_to(v, (3,) + jnp.shape(v)), state)
    actions = jnp.array([c['actions'][1] for c in cases], dtype=jnp.int32)
    _, states, rewards, done, _ = jax.jit(jax.vmap(montezuma_env.step))(states, actions)
    for i, case in enumerate(cases):
        measured = _measured(jax.tree.map(lambda v: v[i], states), done[i])
        assert [int(measured[k]) for k in FIELDS] == case['expected'][1]
        assert float(rewards[i]) == case['rewards'][1]
