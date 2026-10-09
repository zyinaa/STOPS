"""Traces: a system-independent record of one self-evolving run.

A trace stores, for every iteration ("step") of the loop, the paired counts the
detector needs plus the bookkeeping used for reporting (tokens, gate decision,
which artifact was the incumbent). Readers in :mod:`stops.readers` produce traces
from SkillOpt and GEPA run directories; everything downstream reads only traces.

Schema (``stops.trace/v1``)::

    {
      "schema": "stops.trace/v1",
      "meta":  {"system", "run", "benchmark", "model", "n_items", "budget", ...},
      "steps": [
        {"step": 1, "evaluated": true, "n_up": 21, "n_down": 11, "n": 200,
         "accepted": true, "incumbent": "initial", "candidate": "step_0001",
         "tokens": 612345, "flags": []},
        {"step": 40, "evaluated": false, "tokens": 81234, "flags": ["no_candidate"]}
      ],
      "full_run": {"artifact": "best_skill.md", "tokens": 25850000}
    }

Only steps with ``evaluated: true`` are detector rounds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .monitor import Monitor

SCHEMA = "stops.trace/v1"


@dataclass
class Trace:
    meta: dict
    steps: list[dict]
    full_run: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- I/O
    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("schema") != SCHEMA:
            raise ValueError(f"{path}: expected schema {SCHEMA!r}, got {d.get('schema')!r}")
        return cls(meta=d["meta"], steps=d["steps"], full_run=d.get("full_run", {}))

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "meta": self.meta, "steps": self.steps, "full_run": self.full_run}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    # ------------------------------------------------------------- views
    @property
    def name(self) -> str:
        return self.meta.get("name") or self.meta.get("run", "trace")

    def rounds(self) -> list[dict]:
        """Detector rounds: the evaluated steps, in order."""
        return [s for s in self.steps if s.get("evaluated")]

    def total_tokens(self) -> int | None:
        if self.full_run.get("tokens") is not None:
            return int(self.full_run["tokens"])
        toks = [s.get("tokens") for s in self.steps]
        return None if any(t is None for t in toks) else int(sum(toks))

    def tokens_through(self, step) -> int | None:
        """Tokens spent on all steps up to and including ``step``."""
        total = 0
        for s in self.steps:
            if s["step"] > step:
                break
            if s.get("tokens") is None:
                return None
            total += int(s["tokens"])
        return total

    # ------------------------------------------------------------ replay
    def replay(self, eps: float = 0.01, delta: float = 0.05, betting: str = "agrapa", **kw) -> "ReplayResult":
        mon = Monitor(eps=eps, delta=delta, betting=betting, **kw)
        for s in self.rounds():
            mon.update_counts(s["n_up"], s["n_down"], s["n"], incumbent=s.get("incumbent"), step=s["step"])
        return ReplayResult(self, mon)


@dataclass
class ReplayResult:
    trace: Trace
    monitor: Monitor

    @property
    def t_alarm(self) -> int | None:
        return self.monitor.t_alarm

    @property
    def nu_hat(self) -> int | None:
        return self.monitor.nu_hat

    @property
    def alarm_step(self):
        return self.monitor.alarm_step

    @property
    def returned(self) -> Any:
        """theta_{nu_hat - 1} (incumbent label at round nu_hat), or None without alarm."""
        return self.monitor.selected

    @property
    def returned_file(self) -> Any:
        """Exact file of theta_{nu_hat - 1} when the reader records one (SkillOpt)."""
        if self.nu_hat is None:
            return None
        return self.trace.rounds()[self.nu_hat - 1].get("incumbent_file")

    @property
    def tokens_to_alarm(self) -> int | None:
        if self.alarm_step is None:
            return self.trace.total_tokens()
        return self.trace.tokens_through(self.alarm_step)

    @property
    def tokens_saved(self) -> float | None:
        tot, used = self.trace.total_tokens(), self.tokens_to_alarm
        if not tot or used is None:
            return None
        return 1.0 - used / tot

    def flags_through_alarm(self) -> list[str]:
        """Data-quality flags raised on any round up to the alarm (or the end)."""
        last = self.alarm_step
        out: list[str] = []
        for s in self.trace.rounds():
            if last is not None and s["step"] > last:
                break
            for f in s.get("flags", []):
                if f not in out:
                    out.append(f)
        return out

    def as_dict(self) -> dict:
        return {
            "trace": self.trace.name,
            "eps": self.monitor.eps,
            "delta": self.monitor.delta,
            "betting": self.monitor.betting.name,
            "rounds": self.monitor.t,
            "t_alarm": self.t_alarm,
            "alarm_step": self.alarm_step,
            "nu_hat": self.nu_hat,
            "returned": self.returned,
            "returned_file": self.returned_file,
            "tokens_to_alarm": self.tokens_to_alarm,
            "tokens_full": self.trace.total_tokens(),
            "tokens_saved": self.tokens_saved,
            "flags_through_alarm": self.flags_through_alarm(),
        }
