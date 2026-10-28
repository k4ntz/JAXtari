"""Behavioral isolation checks for the focused robustness candidates."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxtari.core import make

PAIRS = {
    "namethisgame": ("shift_tentacle_columns", "infinite_oxygen"),
    "timepilot": ("shift_player_anchor", "no_enemy_fire"),
    "spaceinvaders": ("shift_shield_row", "no_enemy_fire"),
    "beamrider": ("shift_player_row", "no_enemy_shots"),
    "asteroids": ("shift_initial_rocks", "no_ship_collisions"),
    "pong": ("shift_player_plane", "slower_enemy"),
    "breakout": ("shift_paddle_row", "wider_paddle"),
    "freeway": ("shift_car_rows", "no_cars"),
    "phoenix": ("shift_player_row", "no_enemy_fire"),
}


def core(env):
    while hasattr(env, "_env"):
        env = env._env
    return env


def assert_tree_equal(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()


@pytest.mark.parametrize("game,mod", [(g, m) for g, mods in PAIRS.items() for m in mods])
def test_reset_step_render_and_batch(game, mod):
    env = make(game, mods=[mod])
    _, state = env.reset(jax.random.PRNGKey(4))
    obs, next_state, reward, done, _ = env.step(state, jnp.int32(0))
    assert np.isfinite(float(reward))
    assert not bool(done)
    image = np.asarray(env.render(next_state))
    assert image.shape == (210, 160, 3)
    assert image.dtype == np.uint8
    states = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    _, batch, *_ = jax.jit(jax.vmap(env.step, in_axes=(0, None)))(states, jnp.int32(0))
    assert_tree_equal(jax.tree.map(lambda x: x[0], batch), next_state)


@pytest.mark.parametrize("game", PAIRS)
def test_layout_changes_geometry_and_keeps_native_motion_parameters(game):
    vanilla = make(game)
    shifted = make(game, mods=[PAIRS[game][0]])
    key = jax.random.PRNGKey(3)
    vo, vs = vanilla.reset(key)
    mo, ms = shifted.reset(key)
    v, m = core(vanilla), core(shifted)
    if game == "namethisgame":
        np.testing.assert_array_equal(ms.tentacle_base_x, vs.tentacle_base_x + 8)
        assert m.consts.tentacle_base_growth_p == v.consts.tentacle_base_growth_p
        np.testing.assert_array_equal(ms.tentacle_len, vs.tentacle_len)
    elif game == "spaceinvaders":
        np.testing.assert_array_equal(m.consts.BARRICADE_POS[0], v.consts.BARRICADE_POS[0])
        assert m.consts.BARRICADE_POS[1] == v.consts.BARRICADE_POS[1] - 6
        assert_tree_equal(ms, vs)
    elif game == "asteroids":
        active = np.asarray(vs.asteroid_states[:, 3]) != 0
        expected = np.array(vs.asteroid_states)
        expected[active, 1] += 8
        np.testing.assert_array_equal(ms.asteroid_states, expected)
        assert m.consts.ASTEROID_SPEED == v.consts.ASTEROID_SPEED
        assert ms.player_x == vs.player_x and ms.player_y == vs.player_y
    elif game == "freeway":
        np.testing.assert_array_equal(mo.car.y, vo.car.y + 3)
        np.testing.assert_array_equal(ms.cars[:, 0], vs.cars[:, 0])
        np.testing.assert_array_equal(ms.lane_time, vs.lane_time)
        assert_tree_equal(mo.chicken, vo.chicken)
    else:
        axis, delta = {
            "timepilot": ("x", 8), "beamrider": ("y", -4),
            "pong": ("x", -8), "breakout": ("y", -8), "phoenix": ("y", -4),
        }[game]
        assert int(getattr(mo.player, axis)) == int(getattr(vo.player, axis)) + delta
        assert int(mo.player.width) == int(vo.player.width)
        assert int(mo.player.height) == int(vo.player.height)


def test_ntg_infinite_oxygen_prevents_only_oxygen_death():
    v, m = core(make("namethisgame")), core(make("namethisgame", mods=["infinite_oxygen"]))
    _, state = v.reset(jax.random.PRNGKey(0))
    state = state.replace(oxy_bar_px=jnp.int32(0), oxygen_frames_remaining=jnp.int32(0))
    _, native, *_ = v.step(state, jnp.int32(0))
    _, easy, *_ = m.step(state, jnp.int32(0))
    assert int(native.lives_remaining) == int(state.lives_remaining) - 1
    assert int(easy.lives_remaining) == int(state.lives_remaining)
    assert int(easy.oxygen_frames_remaining) == m.consts.hud_bar_initial_px
    assert int(easy.score) == int(state.score)
    # The mod is not invincibility: a bottom-reaching tentacle still kills.
    danger = state.replace(tentacle_len=state.tentacle_len.at[0].set(len(m.consts.tentacle_ys)))
    _, hurt, *_ = m.step(danger, jnp.int32(0))
    assert int(hurt.lives_remaining) == int(state.lives_remaining) - 1


def test_timepilot_no_fire_preserves_rng_and_aircraft():
    v, m = core(make("timepilot")), core(make("timepilot", mods=["no_enemy_fire"]))
    _, state = v.reset(jax.random.PRNGKey(2))
    enemies = state.enemy_states.at[:, 3].set(1)
    missiles = state.enemy_missile_states.at[:, 3].set(0)
    native, nk = v.enemy_missile_step(missiles, enemies, jnp.bool_(True), state.rng_key)
    easy, ek = m.enemy_missile_step(missiles, enemies, jnp.bool_(True), state.rng_key)
    assert np.any(np.asarray(native[:, 3]) > 0)
    assert not np.any(np.asarray(easy[:, 3]) > 0)
    np.testing.assert_array_equal(nk, ek)
    np.testing.assert_array_equal(native[:, :3], easy[:, :3])
    _, native_state, *_ = v.step(state, jnp.int32(0))
    _, easy_state, *_ = m.step(state, jnp.int32(0))
    np.testing.assert_array_equal(native_state.enemy_states, easy_state.enemy_states)
    assert int(native_state.score) == int(easy_state.score)


def test_spaceinvaders_no_fire_retains_targets_and_bunkers():
    v, m = core(make("spaceinvaders")), core(make("spaceinvaders", mods=["no_enemy_fire"]))
    _, state = v.reset(jax.random.PRNGKey(0))
    state = state.replace(enemy_fire_cooldown=jnp.int32(0))
    native = v._update_enemy_bullets(state, jax.random.PRNGKey(2))
    easy = m._update_enemy_bullets(state, jax.random.PRNGKey(2))
    assert bool(jnp.any(native[0]))
    assert not bool(jnp.any(easy[0]))
    for a, b in zip(native[1:], easy[1:], strict=True):
        np.testing.assert_array_equal(a, b)
    _, ns, *_ = v.step(state, jnp.int32(0))
    _, es, *_ = m.step(state, jnp.int32(0))
    np.testing.assert_array_equal(ns.destroyed, es.destroyed)
    np.testing.assert_array_equal(ns.barricade_health, es.barricade_health)
    assert int(ns.player_score) == int(es.player_score)


def test_beamrider_no_shots_does_not_change_ufo_patterns():
    v, m = core(make("beamrider")), core(make("beamrider", mods=["no_enemy_shots"]))
    _, state = v.reset(jax.random.PRNGKey(0))
    level = state.level
    out = m._enemy_shot_step(state, level.white_ufo_pos, level.white_ufo_pattern_id, level.white_ufo_pattern_timer)
    assert not bool(jnp.any(out[0][1] <= m.consts.BOTTOM_CLIP))
    assert int(out[3]) == 0
    # Beamrider donates its state buffers: make the second input independent.
    easy_input = jax.block_until_ready(jax.tree.map(lambda x: x.copy(), state))
    _, native, *_ = v.step(state, jnp.int32(0))
    _, easy, *_ = m.step(easy_input, jnp.int32(0))
    np.testing.assert_array_equal(native.level.white_ufo_pos, easy.level.white_ufo_pos)
    np.testing.assert_array_equal(native.level.white_ufo_pattern_id, easy.level.white_ufo_pattern_id)
    assert int(native.score) == int(easy.score)


def test_asteroids_no_ship_collisions_retains_missile_hits():
    v, m = core(make("asteroids")), core(make("asteroids", mods=["no_ship_collisions"]))
    _, state = v.reset(jax.random.PRNGKey(0))
    rocks = state.asteroid_states.at[:, 3].set(0).at[0].set(jnp.array([80, 100, 1, 1, 0]))
    assert int(v.player_hit(80, 100, True, rocks)) == 0
    assert int(m.player_hit(80, 100, True, rocks)) == -1
    hit_args = (80, 100, True, 0, 0, False, rocks)
    native_hits, easy_hits = v.missile_hit(*hit_args), m.missile_hit(*hit_args)
    assert int(native_hits[0]) == 0
    np.testing.assert_array_equal(native_hits, easy_hits)


def test_pong_slower_enemy_retains_tracking_direction():
    v, m = core(make("pong")), core(make("pong", mods=["slower_enemy"]))
    _, state = v.reset(jax.random.PRNGKey(0))
    state = state.replace(step_counter=jnp.int32(1), enemy_y=jnp.int32(80), ball_y=jnp.int32(140))
    native, easy = v._enemy_step(state), m._enemy_step(state)
    assert int(native.enemy_y) == 82
    assert int(easy.enemy_y) == 81
    assert_tree_equal(native.replace(enemy_y=easy.enemy_y), easy)


def test_breakout_wider_paddle_catches_native_hits_and_additional_hits():
    v, m = core(make("breakout")), core(make("breakout", mods=["wider_paddle"]))
    xs = jnp.arange(0, 160)
    for small in (False, True):
        native = jax.vmap(lambda x: v._detect_paddle_hit(x, 187, 60, small))(xs)
        easy = jax.vmap(lambda x: m._detect_paddle_hit(x, 187, 60, small))(xs)
        assert bool(jnp.all(~native | easy))
        assert int(jnp.sum(easy)) > int(jnp.sum(native))
    # Every native ball speed tier / direction stays unchanged.
    for speed in range(5):
        for direction in range(4):
            assert_tree_equal(v._get_ball_velocity(speed, direction, 3), m._get_ball_velocity(speed, direction, 3))


def test_freeway_no_cars_keeps_crossing_score_and_has_no_phantom_targets():
    env = make("freeway", mods=["no_cars"])
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert not bool(jnp.any(obs.car.active))
    def advance(s, _):
        obs, ns, reward, done, _ = env.step(s, jnp.int32(1))  # UP
        return ns, (reward, obs.car.active)
    final, (rewards, active) = jax.jit(lambda s: jax.lax.scan(advance, s, None, length=400))(state)
    assert float(jnp.sum(rewards)) >= 2
    assert not bool(jnp.any(active))
    assert int(final.score) >= 2


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_phoenix_no_fire_across_all_wave_types(level):
    env = make("phoenix", mods=["no_enemy_fire"])
    c = core(env)
    _, state = env.reset(jax.random.PRNGKey(3))
    state = state.replace(
        level=jnp.int32(level),
        enemies_x=c.consts.ENEMY_POSITIONS_X[level - 1],
        enemies_y=c.consts.ENEMY_POSITIONS_Y[level - 1],
    )
    # Full step path covers both salvo positions and bat/boss fire masks.
    def advance(s, _):
        _, ns, *_ = env.step(s, jnp.int32(0))
        return ns, jnp.any(ns.enemy_projectile_y >= 0)
    _, active = jax.jit(lambda s: jax.lax.scan(advance, s, None, length=240))(state)
    assert not bool(jnp.any(active))


@pytest.mark.parametrize("small", [False, True])
def test_breakout_rendered_paddle_matches_collision_width(small):
    env = make("breakout", mods=["wider_paddle"])
    c = core(env)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(small_paddle=jnp.bool_(small))
    obs = c._get_observation(state)
    image = np.asarray(env.render(state))
    y = int(obs.player.y)
    color = np.asarray(c.consts.PLAYER_COLOR)
    x = int(obs.player.x)
    # Native side walls share the paddle color; inspect its local region.
    region = image[y, x - 2:x + c.consts.PLAYER_SIZE[0] + 2]
    visible_width = int(np.sum(np.all(region == color, axis=-1)))
    assert visible_width == int(obs.player.width)
