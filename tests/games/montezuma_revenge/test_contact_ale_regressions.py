"""Independent native enemy-contact frames.

Position/reflection counterfactuals are explicitly synthetic. Frozen states
check mechanisms; fresh reset and room-start routes establish reachability.
Initial contact oracles precede their fixes. Four additional return routes
were recorded native-first after the final source changes, without fitting.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from test_native_ale_regressions import _measured, _state, native_reset_state, native_step


REFERENCE = json.loads(Path(__file__).with_name('contact_ale_oracles.json').read_text())
FIELDS = REFERENCE['provenance']['fields']


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda c: c['name'])
def test_native_enemy_contact(native_reset_state, native_step, case):
    state = _state(native_reset_state, case['state'])
    assert len(case['actions']) == len(case['expected']) == len(case['rewards'])
    for tick, (action, expected, expected_reward) in enumerate(zip(
            case['actions'], case['expected'], case['rewards'])):
        _, state, reward, done, _ = native_step(state, jnp.int32(action))
        measured = _measured(state, done)
        assert [int(measured[k]) for k in FIELDS] == expected, (case['name'], tick)
        assert float(reward) == expected_reward, (case['name'], tick)
        if expected[FIELDS.index('mode')] == 4:
            assert bool(state.native_player_reflected) == case['native_reflection'][tick], (
                case['name'], tick, 'native B4 bit3')
