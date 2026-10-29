#!/usr/bin/env python3
"""Verify the published inputs and replay the frozen held-out comparison."""

import argparse
import hashlib
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, help="New output directory; never overwrites results"
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--assets-storage",
        type=Path,
        help="Existing JAXtari storage directory containing sprites/",
    )
    parser.add_argument(
        "--games", nargs="+", help="Optional subset; omit for the full paper evaluation"
    )
    parser.add_argument(
        "--recordings-only",
        action="store_true",
        help="Regenerate and verify recordings without running the JAX comparison",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan_bytes = gzip.decompress((root / "frozen_plan.json.gz").read_bytes())
    require(
        hashlib.sha256(plan_bytes).hexdigest()
        == (root / "frozen_plan.sha256").read_text().strip(),
        "Frozen plan hash mismatch",
    )
    plan = json.loads(plan_bytes)
    capsule_manifest = json.loads((root / "capsule_manifest.json").read_text())
    manifest = capsule_manifest["recordings"]
    files = {p.name for p in (root / "capsules").glob("*.pkl.gz")}
    require(
        files == {r["capsule"] for r in manifest},
        "Capsule inventory differs from manifest",
    )
    for row in manifest:
        require(
            digest(root / "capsules" / row["capsule"]) == row["capsule_sha256"],
            f'Capsule hash mismatch: {row["capsule"]}',
        )
    archive = root / "frozen_source.tar.gz"
    require(
        digest(archive) == (root / "frozen_source.sha256").read_text().strip(),
        "Frozen source archive hash mismatch",
    )
    with tempfile.TemporaryDirectory(prefix="jaxtari-heldout-replay-") as temp:
        snapshot = Path(temp)
        with tarfile.open(archive) as tar:
            tar.extractall(snapshot, filter="data")
        expected = {
            name: value
            for name, value in plan["snapshot_files"].items()
            if not name.startswith("assets/")
        }
        actual = {
            str(p.relative_to(snapshot)): digest(p)
            for p in snapshot.rglob("*")
            if p.is_file()
        }
        require(
            actual == expected,
            "Frozen source files differ from the original study inventory",
        )
        print(
            f"Verified {len(manifest)} replay capsules and {len(actual)} frozen source files",
            flush=True,
        )
        if args.verify_only:
            return
        require(
            args.out is not None,
            "Supply --out for replay, or --verify-only to check the published inputs",
        )
        require(
            not args.out.exists(),
            "Output path already exists; choose a new output directory",
        )
        games = args.games or plan["games"]
        require(set(games) <= set(plan["games"]), "Unknown game requested")
        for package, version in plan["packages"].items():
            try:
                installed = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                installed = "missing"
            require(
                installed == version,
                f"{package}: expected {version}, got {installed}; see requirements-replay.txt",
            )
        if args.assets_storage:
            storage = args.assets_storage.resolve()
        else:
            from platformdirs import user_data_dir

            choices = [Path(user_data_dir(name)) for name in ("jaxtari", "jaxatari")]
            storage = next((p for p in choices if (p / "sprites").is_dir()), choices[0])
        # Only the evaluated games' sprite payloads are required. The complete
        # historical asset inventory remains in frozen_plan.json for inspection.
        asset_games = {"montezuma" if g == "montezumarevenge" else g for g in games}
        asset_counts = dict.fromkeys(asset_games, 0)
        for name, value in plan["snapshot_files"].items():
            relative = Path(name)
            if (
                len(relative.parts) < 5
                or relative.parts[:3] != ("assets", "jaxtari", "sprites")
                or relative.parts[3] not in asset_games
                or relative.suffix not in (".npy", ".png")
            ):
                continue
            local = storage.joinpath(*relative.parts[2:])
            require(
                local.is_file() and digest(local) == value,
                f"Missing or changed rendering asset: {local}; install the matching JAXtari assets",
            )
            asset_counts[relative.parts[3]] += 1
        require(
            all(asset_counts.values()),
            "Some requested games have no assets in the frozen manifest",
        )
        asset_count = sum(asset_counts.values())
        (snapshot / "assets").mkdir()
        (snapshot / "assets/jaxtari").symlink_to(storage, target_is_directory=True)
        env = os.environ.copy()
        env.update(
            PYTHONPATH=os.pathsep.join(
                [str(snapshot / "src"), str(snapshot / "scripts")]
            ),
            XDG_DATA_HOME=str(snapshot / "assets"),
            JAX_PLATFORMS="cpu",
            PYTHONDONTWRITEBYTECODE="1",
        )
        command = [
            sys.executable,
            "-B",
            str(snapshot / "scripts/oc_parity/audit_transfers.py"),
            "--corpus",
            str(args.out.resolve()),
            "--out",
            str((args.out / "evaluation").resolve()),
            "--games",
            *games,
            "--n",
            str(plan["compared_frames"]),
            "--num-t0",
            str(plan["syncs_per_recording"]),
            "--seed",
            str(plan["sync_seed"]),
        ]
        regeneration = [
            sys.executable,
            "-B",
            str(root / "regenerate_recordings.py"),
            "--capsules",
            str(root / "capsules"),
            "--out",
            str(args.out.resolve()),
            "--games",
            *games,
        ]
        print(
            f"Verified {asset_count} sprite files; regenerating original recordings from checkpoints and actions",
            flush=True,
        )
        result = subprocess.run(regeneration, env=env)
        if result.returncode:
            raise SystemExit(result.returncode)
        output_plan = dict(plan, games=games)
        output_plan_bytes = (json.dumps(output_plan, indent=2) + "\n").encode()
        (args.out / "plan.json").write_bytes(output_plan_bytes)
        (args.out / "plan.sha256").write_text(
            hashlib.sha256(output_plan_bytes).hexdigest() + "\n"
        )
        recording_rows = []
        for row in manifest:
            if row["game"] not in games:
                continue
            regenerated = {key: row[key] for key in ("file", "game", "seed", "meta")}
            regenerated["sha256"] = digest(args.out / "raw" / row["file"])
            regenerated["original_npz_sha256"] = row["sha256"]
            recording_rows.append(regenerated)
        (args.out / "recordings.json").write_text(
            json.dumps(recording_rows, indent=2) + "\n"
        )
        (args.out / "recording_notes.jsonl").write_text("")
        if args.recordings_only:
            return
        print(
            f"Replaying frozen JAX comparison for {len(games)} games without future fitting",
            flush=True,
        )
        result = subprocess.run(command, env=env)
        (args.out / "evaluation/completion.json").write_text(
            json.dumps({"returncode": result.returncode, "future_assisted": False})
            + "\n"
        )
        if result.returncode:
            raise SystemExit(result.returncode)
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(root / "build_results.py"),
                "--study",
                str(args.out.resolve()),
                "--out",
                str((args.out / "results").resolve()),
            ],
            env=env,
        )
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
