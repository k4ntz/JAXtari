# ALE--JAXtari comparison: reproduction files

This package reproduces the numerical results for the implementation snapshot
frozen on **27 October 2026**. It contains 45 compact checkpoint/action files
(three recordings for each of 15 evaluated games), the frozen source and protocol,
replay scripts, and per-game/per-seed numerical summaries. The source archive
fixes the evaluated implementation independently of the current checkout.

## Requirements

Use Python 3.12, the versions in `requirements-replay.txt`, the matching Atari
ROMs installed for ALE, and the normal JAXtari rendering assets. Install assets
using the [repository setup instructions](../../../../README.md). ROMs and
rendering assets are not bundled. The replay command checks dependency versions,
rendering-asset hashes, and regenerated trajectory-array hashes.

## Verify the package

From the repository root, using standard-library Python only:

```bash
python docs/issue-reports/camera-ready-trajectory-transfer/heldout/reproduce.py --verify-only
```

## Regenerate recordings, measurements, and images

```bash
python docs/issue-reports/camera-ready-trajectory-transfer/heldout/reproduce.py \
  --out data/oc_parity_holdout_reproduction
```

Choose a new output directory; existing directories are refused. Outputs are:

- `raw/`: reconstructed recordings, including actions, RAM, object detections,
  RGB frames, rewards and episode flags.
- `regeneration.json` and `regeneration.log`: reconstruction checks and logs.
- `evaluation/`: frozen per-game measurements and comparison panels.
- `results/`: aggregate JSON, per-window CSV, overview images and `gallery.html`.

Use `--games pong` for a subset or `--recordings-only` to reconstruct the
recordings without evaluating JAXtari. Use `--assets-storage /path/to/jaxtari`
for a custom asset directory containing `sprites/`.

The full evaluation compares 150 frames including injection (149 transitions),
five sampled windows per recording, with synchronization seed 20261027 and no
future-assisted fitting. All expanded outputs are generated locally under the
ignored `data/` directory.

## Included files

- `capsules/` and `capsule_manifest.json`: checkpoint/action inputs and hashes.
- `frozen_source.tar.gz` and `frozen_source.sha256`: evaluated source snapshot.
- `frozen_plan.json.gz` and `frozen_plan.sha256`: protocol, source/asset hashes
  and dependency versions.
- `requirements-replay.txt`: pinned package versions.
- `reproduce.py`, `regenerate_recordings.py`, `build_results.py`: reconstruction,
  evaluation and result-generation entry points.
- `numerical_summary.csv`, `seed_summary.csv`: reference numerical summaries.
