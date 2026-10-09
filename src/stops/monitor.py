"""Online stopping and output selection for a self-evolving loop (Algorithm 1).

Typical use inside any propose / evaluate / gate loop::

    mon = Monitor(eps=0.01, delta=0.05)
    for t in range(1, N + 1):
        cand = propose(inc)
        b, c = evaluate(inc), evaluate(cand)       # per-item 0/1 on D_val
        mon.update(b, c, incumbent=inc)
        if mon.alarm:
            return mon.selected                     # theta_{nu_hat - 1}
        inc = gate(inc, cand)

Rounds are paired evaluations. A loop iteration that produces no candidate is not
a round: do not call :meth:`Monitor.update` for it (pass ``step=`` to keep the
loop's own iteration number alongside the round index).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .betting import BettingScheme, betting_from_config, make_betting


@dataclass
class RoundRecord:
    """What the monitor saw and concluded at one round."""

    t: int                 # round index (1-based count of paired evaluations)
    step: Any              # caller's label for this round, e.g. the loop iteration
    n_up: int              # items the candidate fixed   (X = +1)
    n_down: int            # items the candidate broke   (X = -1)
    n: int                 # items compared
    z: float               # Z_t = (n_up - n_down) / n
    log_m: float           # log M_t = log max_s W_t^(s)
    s_star: int            # argmax_s W_t^(s), the current change-point estimate
    alarm: bool            # True only on the round where M_t first reaches 1/delta


@dataclass
class _Restart:
    start: int
    logw: list[float]
    state: dict = field(default_factory=dict)


def paired_counts(baseline, candidate) -> tuple[int, int, int]:
    """Count (fixed, broken, compared) from per-item 0/1 outcomes.

    ``baseline`` and ``candidate`` are either two equal-length sequences aligned by
    position, or two mappings ``item_id -> 0/1`` compared on their common ids.
    """
    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        keys = baseline.keys() & candidate.keys()
        if not keys:
            raise ValueError("baseline and candidate share no item ids")
        pairs = ((baseline[k], candidate[k]) for k in keys)
    elif isinstance(baseline, Mapping) or isinstance(candidate, Mapping):
        raise TypeError("pass two mappings or two sequences, not one of each")
    else:
        if len(baseline) != len(candidate):
            raise ValueError(f"length mismatch: {len(baseline)} vs {len(candidate)}")
        if len(baseline) == 0:
            raise ValueError("no items to compare")
        pairs = zip(baseline, candidate)
    up = down = n = 0
    for b, c in pairs:
        b, c = _as_binary(b), _as_binary(c)
        n += 1
        if c > b:
            up += 1
        elif c < b:
            down += 1
    return up, down, n


def _as_binary(v) -> int:
    if v in (0, 1):  # also accepts True/False and 0.0/1.0
        return int(v)
    raise ValueError(f"per-item outcomes must be binary 0/1, got {v!r}")


def _logmeanexp(xs: Sequence[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs) / len(xs))


class Monitor:
    """Restart e-detector with change-point based output selection.

    Parameters
    ----------
    eps
        Worthwhile-gain threshold. The pre-change condition is
        ``E[Z_t | F_{t-1}] >= eps``.
    delta
        False-alarm control. The alarm fires when ``M_t >= 1/delta``; under the
        pre-change condition the average run length to a false alarm is at least
        ``1/delta``.
    betting
        ``"agrapa"`` (default) or ``"mixture"``, or a :class:`BettingScheme`.
    lambdas, lam0, lam_max
        Options for the betting scheme (see :mod:`stops.betting`).
    """

    def __init__(
        self,
        eps: float = 0.01,
        delta: float = 0.05,
        betting: str | BettingScheme = "agrapa",
        *,
        lambdas: Sequence[float] | None = None,
        lam0: float | None = None,
        lam_max: float | None = None,
    ):
        if not 0.0 < eps < 1.0:
            raise ValueError(f"eps must be in (0, 1), got {eps}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.eps = float(eps)
        self.delta = float(delta)
        self.betting = make_betting(betting, lambdas=lambdas, lam0=lam0, lam_max=lam_max)
        self.history: list[RoundRecord] = []
        self._restarts: list[_Restart] = []
        self._incumbents: list[Any] = []
        self.t_alarm: int | None = None
        self.nu_hat: int | None = None

    # ------------------------------------------------------------------ updates
    def update(self, baseline, candidate, *, incumbent: Any = None, step: Any = None) -> bool:
        """Feed one round of per-item outcomes. Returns ``True`` once the alarm has fired.

        ``baseline`` holds the incumbent's outcomes (theta_{t-1}) and ``candidate``
        the proposal's outcomes, both on the same validation items. ``incumbent`` is
        whatever identifies theta_{t-1} for you (an object, a path, a hash); it is
        what :attr:`selected` returns.
        """
        n_up, n_down, n = paired_counts(baseline, candidate)
        return self.update_counts(n_up, n_down, n, incumbent=incumbent, step=step)

    def update_counts(
        self, n_up: int, n_down: int, n: int, *, incumbent: Any = None, step: Any = None
    ) -> bool:
        """Same as :meth:`update`, from the round's sufficient statistics."""
        n_up, n_down, n = int(n_up), int(n_down), int(n)
        if n <= 0 or n_up < 0 or n_down < 0 or n_up + n_down > n:
            raise ValueError(f"invalid counts: n_up={n_up}, n_down={n_down}, n={n}")

        t = len(self.history) + 1
        self._incumbents.append(incumbent)
        self._restarts.append(
            _Restart(start=t, logw=[0.0] * self.betting.n_components, state=self.betting.init_state())
        )

        eps = self.eps
        n_zero = n - n_up - n_down
        z = (n_up - n_down) / n
        q = (n_up + n_down) / n
        for r in self._restarts:
            lams = self.betting.fractions(r.state, eps)
            for k, lam in enumerate(lams):
                r.logw[k] += (
                    n_up * math.log1p(lam * (eps - 1.0))
                    + n_down * math.log1p(lam * (eps + 1.0))
                    + n_zero * math.log1p(lam * eps)
                )
            self.betting.observe(r.state, z, q)

        log_m, s_star = -math.inf, 0
        for r in self._restarts:  # first maximiser wins ties (earliest start)
            lw = _logmeanexp(r.logw)
            if lw > log_m:
                log_m, s_star = lw, r.start

        fired = self.t_alarm is None and log_m >= -math.log(self.delta)
        if fired:
            self.t_alarm, self.nu_hat = t, s_star
        self.history.append(RoundRecord(t, step, n_up, n_down, n, z, log_m, s_star, fired))
        return self.alarm

    # --------------------------------------------------------------- read-outs
    @property
    def t(self) -> int:
        """Rounds observed so far."""
        return len(self.history)

    @property
    def alarm(self) -> bool:
        return self.t_alarm is not None

    @property
    def threshold(self) -> float:
        return 1.0 / self.delta

    @property
    def wealth(self) -> float:
        """Current M_t (1.0 before any round)."""
        return math.exp(self.history[-1].log_m) if self.history else 1.0

    @property
    def selected(self) -> Any:
        """theta_{nu_hat - 1}: the incumbent that round ``nu_hat`` was compared against.

        ``None`` before the alarm. Equals the ``incumbent`` passed at round ``nu_hat``.
        """
        if self.nu_hat is None:
            return None
        return self._incumbents[self.nu_hat - 1]

    @property
    def alarm_step(self) -> Any:
        """The caller's ``step`` label of the alarm round."""
        return None if self.t_alarm is None else self.history[self.t_alarm - 1].step

    @property
    def nu_hat_step(self) -> Any:
        """The caller's ``step`` label of round ``nu_hat``."""
        return None if self.nu_hat is None else self.history[self.nu_hat - 1].step

    def restart_log_wealth(self) -> dict[int, float]:
        """log W_t^(s) for every restart s at the latest round."""
        return {r.start: _logmeanexp(r.logw) for r in self._restarts}

    def summary(self) -> str:
        head = (
            f"eps={self.eps} delta={self.delta} threshold={self.threshold:g} "
            f"betting={self.betting.name} rounds={self.t}"
        )
        if not self.alarm:
            return head + "  no alarm"
        return head + f"  alarm at round {self.t_alarm} (step {self.alarm_step}), nu_hat={self.nu_hat}"

    # ------------------------------------------------------------- persistence
    def config(self) -> dict:
        return {"eps": self.eps, "delta": self.delta, "betting": self.betting.config()}

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "config": self.config(),
            "history": [asdict(r) for r in self.history],
            "restarts": [asdict(r) for r in self._restarts],
            "incumbents": self._incumbents,
            "t_alarm": self.t_alarm,
            "nu_hat": self.nu_hat,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "Monitor":
        cfg = d["config"]
        mon = cls(cfg["eps"], cfg["delta"], betting_from_config(cfg["betting"]))
        mon.history = [RoundRecord(**r) for r in d["history"]]
        mon._restarts = [_Restart(**r) for r in d["restarts"]]
        mon._incumbents = list(d["incumbents"])
        mon.t_alarm, mon.nu_hat = d["t_alarm"], d["nu_hat"]
        return mon

    def save(self, path: str | Path) -> None:
        """Write the full state as JSON (incumbents must be JSON-serialisable)."""
        Path(path).write_text(json.dumps(self.state_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Monitor":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ replay
    @classmethod
    def replay(
        cls,
        rounds: Iterable[Sequence[int] | Mapping[str, Any]],
        *,
        stop_at_alarm: bool = False,
        **kwargs,
    ) -> "Monitor":
        """Run a fresh monitor over recorded rounds.

        Each round is ``(n_up, n_down, n)`` or a mapping with keys ``n_up``,
        ``n_down``, ``n`` and optionally ``incumbent`` and ``step``.
        """
        mon = cls(**kwargs)
        for r in rounds:
            if isinstance(r, Mapping):
                mon.update_counts(
                    r["n_up"], r["n_down"], r["n"], incumbent=r.get("incumbent"), step=r.get("step")
                )
            else:
                mon.update_counts(*r)
            if stop_at_alarm and mon.alarm:
                break
        return mon
