"""Read a SkillOpt run directory into a :class:`~stops.trace.Trace`.

Needs the run's ``history.json`` and, for every evaluated step,
``steps/step_XXXX/paired_outcomes.json`` (written by the instrumentation patch in
``integrations/skillopt``). ``config.json`` and ``runtime_state.json`` are used when
present.

Conventions (see docs/conventions.md):

* A step whose gate action is ``skip_*`` produced no candidate. It is kept in the
  trace with ``evaluated: false`` and is not a detector round.
* If a step's paired file has no common items (SkillOpt answered the candidate
  from its score cache without per-item outcomes), the round is recorded as
  ``n`` ties with flag ``missing_pairs``.
* SkillOpt's slow update rewrites the incumbent at epoch ends without refreshing
  the per-item baseline used in ``paired_outcomes.json``. Rounds whose baseline
  predates such a rewrite get flag ``baseline_stale`` (or
  ``baseline_possibly_stale`` when the history cannot tell).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..trace import Trace

_SLOW_RE = re.compile(r"^slow_update_epoch_(\d+)$")


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _step_tokens(rec: dict) -> int:
    total = 0
    for v in (rec.get("tokens") or {}).values():
        if isinstance(v, dict):
            total += int(v.get("prompt_tokens", 0)) + int(v.get("completion_tokens", 0))
    return total


def _binary(v) -> int:
    return int(round(float(v)))


def _paired(path: Path) -> tuple[int, int, int]:
    d = _read_json(path)
    b = d.get("baseline_per_item") or {}
    c = d.get("candidate_per_item") or {}
    common = b.keys() & c.keys()
    up = down = 0
    for k in common:
        bi, ci = _binary(b[k]["hard"]), _binary(c[k]["hard"])
        if ci > bi:
            up += 1
        elif ci < bi:
            down += 1
    return up, down, len(common)


def _stale_flags(history: list[dict], use_slow: bool) -> dict[int, str]:
    """Map step -> 'baseline_stale' / 'baseline_possibly_stale' for affected rounds.

    A real slow update happens after epoch e >= 2 when it produced content; the
    history shows it as current_origin == 'slow_update_epoch_XX' on later rejected
    steps. If the first step after epoch e was accepted the label is overwritten
    and we cannot tell, so the rounds it could affect are 'possibly' stale.
    """
    if not use_slow or not history:
        return {}
    epoch_last: dict[int, int] = {}
    for h in history:
        epoch_last[int(h["epoch"])] = int(h["step"])
    real = set()
    for h in history:
        m = _SLOW_RE.match(str(h.get("current_origin") or ""))
        if m:
            real.add(int(m.group(1)))
    unknown = set()
    epochs = sorted(epoch_last)
    for e in epochs:
        if e < 2 or e in real or e == epochs[-1]:
            continue
        nxt = [h for h in history if int(h["epoch"]) == e + 1 and not str(h.get("action", "")).startswith("skip")]
        if nxt and str(nxt[0].get("action", "")).startswith("accept"):
            unknown.add(e)

    out: dict[int, str] = {}
    last_accept = 0
    for h in history:
        step = int(h["step"])
        # epochs that ended after the incumbent was last set and before this step
        hits = [e for e, last in epoch_last.items() if last_accept <= last < step]
        if any(e in real for e in hits):
            out[step] = "baseline_stale"
        elif any(e in unknown for e in hits):
            out[step] = "baseline_possibly_stale"
        if str(h.get("action", "")).startswith("accept"):
            last_accept = step
    return out


def read_skillopt(run_dir: str | Path, *, name: str | None = None) -> Trace:
    run_dir = Path(run_dir)
    history = _read_json(run_dir / "history.json")
    if isinstance(history, dict):
        history = [history]
    cfg = _read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    rs = _read_json(run_dir / "runtime_state.json") if (run_dir / "runtime_state.json").exists() else {}

    stale = _stale_flags(history, bool(cfg.get("use_slow_update", False)))

    # n_items: the size of a fully paired round
    paired_files = {
        int(h["step"]): run_dir / "steps" / f"step_{int(h['step']):04d}" / "paired_outcomes.json" for h in history
    }
    counts = {s: _paired(p) for s, p in paired_files.items() if p.exists()}
    n_items = max((n for _, _, n in counts.values()), default=0)
    if n_items == 0:
        raise ValueError(f"{run_dir}: no paired_outcomes.json with per-item results")

    steps = []
    incumbent = "initial"
    prev_step = 0
    for h in history:
        step = int(h["step"])
        action = str(h.get("action", ""))
        rec = {"step": step, "epoch": h.get("epoch"), "action": action, "tokens": _step_tokens(h)}
        flags = []
        if step in counts:
            up, down, n = counts[step]
            if n == 0:
                up, down, n = 0, 0, n_items
                flags.append("missing_pairs")
            rec.update(evaluated=True, n_up=up, n_down=down, n=n)
        else:
            rec["evaluated"] = False
            flags.append("no_candidate")
        rec["accepted"] = action.startswith("accept")
        rec["incumbent"] = incumbent
        # SkillOpt saves the live skill after every step (and after slow updates)
        rec["incumbent_file"] = f"skills/skill_v{prev_step:04d}.md" if prev_step else "initial"
        rec["candidate"] = f"step_{step:04d}"
        rec["candidate_score"] = h.get("selection_hard")  # candidate's validation accuracy
        rec["incumbent_score"] = h.get("current_score")   # incumbent's validation accuracy after the gate
        if step in stale and rec["evaluated"]:
            flags.append(stale[step])
        rec["flags"] = flags
        steps.append(rec)
        prev_step = step
        if rec["accepted"]:
            incumbent = f"step_{step:04d}"

    meta = {
        "name": name or run_dir.name,
        "system": "skillopt",
        "run": run_dir.name,
        "benchmark": cfg.get("env"),
        "model": cfg.get("target_model"),
        "n_items": n_items,
        "budget": len(history),
        "steps_per_epoch": cfg.get("steps_per_epoch"),
        "use_slow_update": cfg.get("use_slow_update"),
        "slow_update_gate_with_selection": cfg.get("slow_update_gate_with_selection"),
        "artifact_paths": {
            "initial": "the run's initial skill",
            "step_XXXX": "steps/step_XXXX/candidate_skill.md",
        },
    }
    full_run = {
        "artifact": "best_skill.md",
        "best_origin": rs.get("best_origin"),
        "tokens": sum(s["tokens"] for s in steps),
    }
    return Trace(meta=meta, steps=steps, full_run=full_run)
