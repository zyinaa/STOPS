"""Rebuild traces/*.json from raw run directories (authors only).

Users do not need this: the traces are committed. Run it after re-running an
experiment, then `pytest` to see whether the paper numbers still hold.

    python scripts/build_traces.py \
        --skillopt-outputs /path/to/SkillOpt/outputs \
        --gepa-runs /path/to/seas-cross-system/runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stops.readers import read_gepa, read_skillopt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skillopt-outputs", type=Path)
    ap.add_argument("--gepa-runs", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "traces")
    a = ap.parse_args()

    cells = json.loads((ROOT / "paper" / "cells.json").read_text(encoding="utf-8"))["cells"]
    for c in cells:
        if c["system"] == "skillopt":
            if not a.skillopt_outputs:
                continue
            tr = read_skillopt(a.skillopt_outputs / c["run"], name=c["id"])
        else:
            if not a.gepa_runs:
                continue
            tr = read_gepa(a.gepa_runs / c["run"], tokens_per_call=c.get("tokens_per_call"), name=c["id"])
        tr.meta["benchmark"] = c["benchmark"]
        tr.meta["model"] = c["model"]
        out = a.out / f"{c['id']}.json"
        tr.save(out)
        print(f"{c['id']:36s} rounds={len(tr.rounds()):>3} steps={len(tr.steps):>3} n={tr.meta['n_items']}")


if __name__ == "__main__":
    main()
