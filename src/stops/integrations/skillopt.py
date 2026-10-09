"""Glue between SkillOpt's trainer loop and :class:`stops.Monitor`.

The trainer patch in ``integrations/skillopt/`` calls three methods:

* :meth:`SkillOptHook.on_round` right after a step's paired outcomes are written,
  with the incumbent's skill text (theta_{t-1}).
* :attr:`SkillOptHook.should_stop` after each step; the trainer breaks out of its
  loops when it is true.
* :meth:`SkillOptHook.finalize` before the trainer writes ``best_skill.md``.

Everything is stored under ``<out_root>/stops/``: the monitor state (so a resumed
run continues where it left off), one snapshot of every incumbent the monitor
saw, ``rounds.jsonl`` for live watching, and ``result.json`` at the end.

Config (YAML ``stops:`` block, all optional)::

    stops:
      enabled: true
      eps: 0.01
      delta: 0.05
      betting: agrapa        # or mixture
      stop: true             # false = monitor and log only, never stop
      refresh_baseline: true # re-evaluate the incumbent after a slow update
      replace_best: true     # on alarm, best_skill.md := theta_{nu_hat-1}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ..monitor import Monitor

DEFAULTS = {
    "enabled": False,
    "eps": 0.01,
    "delta": 0.05,
    "betting": "agrapa",
    "stop": True,
    "refresh_baseline": True,
    "replace_best": True,
}


def hook_config(cfg: Mapping) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg.get("stops") or {}).items() if k in DEFAULTS})
    return out


def _hard(rec) -> int:
    v = rec["hard"] if isinstance(rec, Mapping) else rec
    return int(round(float(v)))


def per_item_counts(baseline: Mapping, candidate: Mapping) -> tuple[int, int, int, bool]:
    """(n_up, n_down, n, missing) from SkillOpt per-item dicts ``id -> {"hard": ...}``.

    When the two dicts share no ids (SkillOpt answered the candidate from its score
    cache), the round counts as ties over the larger dict, and ``missing`` is True.
    """
    common = baseline.keys() & candidate.keys()
    if not common:
        return 0, 0, max(len(baseline), len(candidate)), True
    up = down = 0
    for k in common:
        b, c = _hard(baseline[k]), _hard(candidate[k])
        if c > b:
            up += 1
        elif c < b:
            down += 1
    return up, down, len(common), False


class SkillOptHook:
    def __init__(self, cfg: Mapping, out_root: str | Path):
        self.cfg = hook_config(cfg)
        self.out_root = Path(out_root)
        self.dir = self.out_root / "stops"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "incumbents").mkdir(exist_ok=True)
        self._state = self.dir / "monitor.json"
        if self._state.exists():
            self.monitor = Monitor.load(self._state)
        else:
            self.monitor = Monitor(eps=self.cfg["eps"], delta=self.cfg["delta"], betting=self.cfg["betting"])

    @property
    def enabled(self) -> bool:
        return bool(self.cfg["enabled"])

    @property
    def refresh_baseline(self) -> bool:
        return bool(self.cfg["refresh_baseline"])

    @property
    def should_stop(self) -> bool:
        return self.enabled and bool(self.cfg["stop"]) and self.monitor.alarm

    def on_round(self, step: int, baseline_per_item: Mapping, candidate_per_item: Mapping, incumbent_skill: str) -> bool:
        """Feed one SkillOpt step. Idempotent per step (safe across resumes)."""
        if any(r.step == step for r in self.monitor.history):
            return self.monitor.alarm
        if not baseline_per_item and not candidate_per_item:
            return self.monitor.alarm  # nothing evaluated this step
        up, down, n, missing = per_item_counts(baseline_per_item, candidate_per_item)
        t = self.monitor.t + 1
        snap = self.dir / "incumbents" / f"round_{t:04d}_step_{step:04d}.md"
        snap.write_text(incumbent_skill, encoding="utf-8")
        rel = snap.relative_to(self.out_root).as_posix()
        self.monitor.update_counts(up, down, n, incumbent=rel, step=step)
        self.monitor.save(self._state)
        r = self.monitor.history[-1]
        line = {"t": r.t, "step": step, "n_up": up, "n_down": down, "n": n, "z": r.z,
                "log_m": r.log_m, "s_star": r.s_star, "alarm": r.alarm, "missing_pairs": missing}
        with open(self.dir / "rounds.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        if r.alarm:
            print(f"    [stops] alarm at round {r.t} (step {step}); nu_hat={self.monitor.nu_hat}; "
                  f"return {self.monitor.selected}", flush=True)
        return self.monitor.alarm

    def selected_skill(self) -> str | None:
        sel = self.monitor.selected
        return None if sel is None else (self.out_root / sel).read_text(encoding="utf-8")

    def finalize(self) -> dict:
        m = self.monitor
        res = {
            "config": self.cfg,
            "rounds": m.t,
            "alarm": m.alarm,
            "t_alarm": m.t_alarm,
            "alarm_step": m.alarm_step,
            "nu_hat": m.nu_hat,
            "nu_hat_step": m.nu_hat_step,
            "selected": m.selected,
            "stopped_early": self.should_stop,
        }
        (self.dir / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        return res
