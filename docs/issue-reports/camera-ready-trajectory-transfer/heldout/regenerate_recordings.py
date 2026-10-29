"""Expand compact replay capsules using the frozen ALE/OCAtari environment.

Invoked by reproduce.py with the frozen source on PYTHONPATH. A capsule stores
one reconstructed emulator checkpoint, the short original prefix needed at that
checkpoint, decoder state, actions, and semantic hashes of every original array.
It is validated for the recorded action sequence, not arbitrary counterfactuals.
"""

import argparse
import contextlib
from enum import Enum
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def array_digest(array):
    array = np.asarray(array)
    h = hashlib.sha256()
    h.update(array.dtype.str.encode())
    h.update(json.dumps(array.shape).encode())
    if array.dtype.hasobject:

        def scalar(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, Enum):
                return value.value
            raise TypeError(f"Unsupported object-array scalar: {type(value).__name__}")

        h.update(
            json.dumps(array.tolist(), separators=(",", ":"), default=scalar).encode()
        )
    else:
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


def restore_wrapper_steps(env, values):
    wrapper = env._env
    index = 0
    while wrapper is not None:
        if str(index) in values:
            wrapper._elapsed_steps = values[str(index)]
        index += 1
        wrapper = getattr(wrapper, "env", None)


def expand(path, output):
    import ale_py
    from oc_parity.record_ocatari import make_oc_env, _snapshot_step
    from oc_parity.trajectory_io import _arrays_to_objects, save_trajectory

    capsule = pickle.loads(gzip.decompress(path.read_bytes()))
    meta, prefix = capsule["meta"], capsule["prefix"]
    env = make_oc_env(meta["game"], meta["seed"])
    try:
        state = ale_py.ALEState.__new__(ale_py.ALEState)
        state.__setstate__((capsule["emulator_state"],))
        env._env.unwrapped.ale.restoreSystemState(state)
        restore_wrapper_steps(env, capsule["wrapper_steps"])
        env.objects = pickle.loads(capsule["decoder_state"])
        objects = _arrays_to_objects(prefix)
        frames = list(prefix["frames"])
        rams = list(prefix["ram"])
        rewards = prefix["rewards"].tolist()
        terminated = prefix["terminated"].tolist()
        truncated = prefix["truncated"].tolist()
        actions = np.asarray(capsule["actions"], dtype=np.int32)
        for t in range(len(frames), len(actions)):
            if terminated[-1] or truncated[-1]:
                env.reset()
            _, reward, term, trunc, ram, obs, frame = _snapshot_step(
                env, int(actions[t]), True
            )
            frames.append(frame)
            rams.append(ram)
            objects.append(obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
        save_trajectory(
            str(output),
            meta=meta,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            ram=rams,
            objects=objects,
            frames=frames,
        )
        with np.load(output, allow_pickle=True) as regenerated:
            actual = {key: array_digest(regenerated[key]) for key in regenerated.files}
        if actual != capsule["array_sha256"]:
            mismatches = [
                key
                for key in set(actual) | set(capsule["array_sha256"])
                if actual.get(key) != capsule["array_sha256"].get(key)
            ]
            raise RuntimeError(
                f"Regeneration differs from original arrays: {mismatches}"
            )
        return {
            "file": output.name,
            "all_arrays_match_original": True,
            "prefix_frames": len(prefix["frames"]),
            "original_npz_sha256": capsule["original_npz_sha256"],
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsules", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", nargs="+")
    args = parser.parse_args()
    paths = sorted(args.capsules.glob("*.pkl.gz"))
    if args.games:
        paths = [p for p in paths if p.name.split("_play_")[0] in args.games]
    if not paths or args.out.exists():
        raise SystemExit("Need replay capsules and a new output directory")
    (args.out / "raw").mkdir(parents=True)
    reports = []
    for path in paths:
        name = path.name.removesuffix(".pkl.gz") + ".npz"
        with (
            (args.out / "regeneration.log").open("a") as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            report = expand(path, args.out / "raw" / name)
        reports.append(report)
        print(json.dumps(report), flush=True)
        (args.out / "regeneration.json").write_text(
            json.dumps(reports, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
