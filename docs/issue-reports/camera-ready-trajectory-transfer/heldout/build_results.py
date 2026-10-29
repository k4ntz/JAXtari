"""Publish a completed frozen evaluation without modifying its measurements.

Usage: python scripts/oc_parity/summarize_heldout.py --study STUDY --out OUT
Threshold counts are post-hoc descriptive diagnostics, not exclusion rules.
"""

import argparse
import csv
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw


def stats(runs, phase, key):
    values = [r[f"metrics_at_{phase}"][key] for r in runs]
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return {
        "mean": float(np.mean(finite)) if finite else None,
        "median": float(np.median(finite)) if finite else None,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "finite_n": len(finite),
        "total_n": len(values),
    }


def acceptable(run, phase, threshold):
    m = run[f"metrics_at_{phase}"]
    return (
        m["entity_mean_l1"] is not None
        and m["entity_mean_l1"] <= threshold
        and m["entity_presence"] >= 0.95
    )


def describe(runs):
    return {
        "n": len(runs),
        "initial_within_2px_and_95pct_presence": sum(
            acceptable(r, "0", 2) for r in runs
        ),
        "endpoint_within_12px_and_95pct_presence": sum(
            acceptable(r, "end", 12) for r in runs
        ),
        "no_persistent_soft_divergence": sum(
            r["first_diverge_soft"] is None for r in runs
        ),
        **{
            f"{key}@{phase}": stats(runs, phase, key)
            for phase in ("0", "end")
            for key in ("entity_mean_l1", "entity_presence", "pixel_mae")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    study, out = args.study.resolve(), args.out.resolve()
    plan = json.loads((study / "plan.json").read_text())
    completion = json.loads((study / "evaluation/completion.json").read_text())
    if completion["returncode"]:
        raise SystemExit("Evaluation has failures; inspect original logs first")
    out.mkdir(parents=True, exist_ok=True)
    rows, summaries, cards = [], {}, []
    contact = {seed: [] for seed in plan["recording_seeds"]}
    for game in plan["games"]:
        source = study / "evaluation" / game
        summary = json.loads((source / "summary.json").read_text())
        summaries[game] = {"all": describe(summary["runs"]), "seeds": {}}
        target = out / game
        target.mkdir(exist_ok=True)
        shutil.copy2(source / "summary.json", target / "summary.json")
        for seed in plan["recording_seeds"]:
            runs = [r for r in summary["runs"] if f"seed{seed:02d}" in r["traj"]]
            summaries[game]["seeds"][str(seed)] = describe(runs)
            first = min(runs, key=lambda r: r["t0"])
            image = Image.open(source / first["panel"]).convert("RGB")
            tile = Image.new("RGB", (480, 446), "white")
            tile.paste(image, (0, 26))
            ImageDraw.Draw(tile).text(
                (4, 6),
                f"{game}: seed {seed}, t0={first['t0']}; ALE | JAX | diff",
                fill="black",
            )
            contact[seed].append(tile)
        cards.append(f"<h2>{html.escape(game)}</h2>")
        for r in summary["runs"]:
            shutil.copy2(source / r["panel"], target / r["panel"])
            seed = int(Path(r["traj"]).stem.rsplit("seed", 1)[1])
            row = {
                "game": game,
                "seed": seed,
                "t0": r["t0"],
                "first_diverge_soft": r["first_diverge_soft"],
            }
            for phase in ("0", "end"):
                for key in (
                    "entity_mean_l1",
                    "entity_presence",
                    "pixel_mae",
                    "entity_matched",
                    "entity_missing",
                ):
                    if key in ("entity_matched", "entity_missing"):
                        entities = r[f"per_entity_at_{phase}"]
                        matched = sum(
                            bool(e["present_both"]) for e in entities.values()
                        )
                        row[f"{key}@{phase}"] = (
                            matched
                            if key == "entity_matched"
                            else len(entities) - matched
                        )
                    else:
                        row[f"{key}@{phase}"] = r[f"metrics_at_{phase}"][key]
            rows.append(row)
            cards.append(
                f'<details><summary>Seed {seed}, t0={r["t0"]}: '
                f'L1 {row["entity_mean_l1@0"]} → {row["entity_mean_l1@end"]}; '
                f'presence {row["entity_presence@0"]:.1%} → {row["entity_presence@end"]:.1%}'
                f'</summary><img src="{game}/{r["panel"]}" alt="ALE, JAXtari, difference at injection and endpoint"></details>'
            )
    with (out / "windows.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "aggregate.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False) + "\n"
    )
    for seed, tiles in contact.items():
        sheet = Image.new("RGB", (480 * 5, 446 * 3), "white")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (i % 5 * 480, i // 5 * 446))
        sheet.save(out / f"seed{seed:02d}_overview.png")
    (out / "gallery.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Frozen held-out transfer evaluation</title>'
        "<style>body{font:16px sans-serif;max-width:1000px;margin:32px auto}summary{padding:8px;cursor:pointer}img{image-rendering:pixelated;width:720px;max-width:100%}</style>"
        "<h1>Held-out transfer: all sampled windows</h1><p>Each panel: ALE | JAXtari | absolute RGB difference. "
        "Top: injection. Bottom: 149 transitions later. No future fitting or replay corrections.</p>"
        + "\n".join(cards)
    )
    for name in ("recordings.json", "recording_notes.jsonl", "plan.sha256"):
        shutil.copy2(study / name, out / name)
    (out / "protocol.json").write_text(
        json.dumps({k: v for k, v in plan.items() if k != "snapshot_files"}, indent=2)
        + "\n"
    )
    for game, d in summaries.items():
        a = d["all"]
        print(
            game,
            a["n"],
            "L1",
            round(a["entity_mean_l1@0"]["mean"], 3),
            round(a["entity_mean_l1@end"]["mean"], 2),
            "presence",
            round(a["entity_presence@0"]["mean"], 3),
            round(a["entity_presence@end"]["mean"], 3),
            "endpoint diagnostic",
            a["endpoint_within_12px_and_95pct_presence"],
        )


if __name__ == "__main__":
    main()
