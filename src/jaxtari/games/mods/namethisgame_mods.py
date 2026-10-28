from jaxtari.modification import JaxtariModController
from jaxtari.games.mods.namethisgame.namethisgame_mod_plugins import (
    NTGShiftColumnsMod,
    NTGInfiniteOxygenMod,
)


class NameThisGameEnvMod(JaxtariModController):
    """Game-specific mod controller for Name This Game."""

    REGISTRY = {
        "shift_tentacle_columns": NTGShiftColumnsMod,
        "infinite_oxygen": NTGInfiniteOxygenMod,
    }

    def __init__(self, env, mods_config: list = [], allow_conflicts: bool = False):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
