from functools import partial

import jax
import jax.numpy as jnp

from jaxtari.modification import JaxtariInternalModPlugin


class NTGShiftColumnsMod(JaxtariInternalModPlugin):
    """Translate all eight tentacle bases right by 8px; retain all rules."""
    constants_overrides = {
        "tentacle_base_x": jnp.arange(24, 144, 16, dtype=jnp.int32),
    }


class NTGInfiniteOxygenMod(JaxtariInternalModPlugin):
    """Remove oxygen pressure; retain shark/tentacle tasks and their scores."""
    @partial(jax.jit, static_argnums=(0,))
    def _update_oxygen(self, state):
        full = jnp.asarray(self._env.consts.hud_bar_initial_px, jnp.int32)
        # Preserve the ordinary contact counter and any other oxygen effects.
        out = type(self._env)._update_oxygen(self._env, state.replace(oxy_bar_px=full))
        return out.replace(oxy_bar_px=full, oxygen_frames_remaining=full)
