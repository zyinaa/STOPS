"""Betting fractions for the restart wealth processes.

Each restart (a gambler that enters at round ``s``) bets on the paired outcomes
``X in {-1, 0, +1}`` with a fraction ``lambda`` that must be predictable, that is,
fixed before the round's outcomes are seen, and lie in ``[0, 1/2]``.

Two schemes are provided, matching Appendix B of the paper:

* :class:`AGRAPA`  the growth-optimal plug-in, one fraction per gambler,
  estimated from the gambler's own window ``[s, t-1]``.
* :class:`FixedMixture`  one wealth process per fixed fraction in a grid,
  averaged with uniform weights.
"""

from __future__ import annotations

from typing import Iterable, Sequence

LAMBDA_CAP = 0.5


class BettingScheme:
    """Interface. A scheme owns no wealth; it only proposes fractions."""

    name: str = "base"

    @property
    def n_components(self) -> int:  # number of parallel wealth processes per gambler
        raise NotImplementedError

    def init_state(self) -> dict:
        """Per-gambler sufficient statistics, created when the gambler enters."""
        return {}

    def fractions(self, state: dict, eps: float) -> list[float]:
        """Fractions for the coming round, computed from rounds already observed."""
        raise NotImplementedError

    def observe(self, state: dict, z: float, q: float) -> None:
        """Fold in one finished round: ``z`` = mean of X, ``q`` = discordant rate."""

    def config(self) -> dict:
        raise NotImplementedError


class AGRAPA(BettingScheme):
    """Adaptive growth-optimal plug-in (Waudby-Smith and Ramdas, 2024).

    For a gambler that entered at round ``s``, at round ``t``::

        mu   = eps - mean(Z_u),             u in [s, t-1]
        var  = mean(q_u) - mean(Z_u)^2
        lam  = min(max(mu, 0) / (var + mu^2), lam_max)

    and ``lam = lam0`` on the entry round, when the window is empty.
    """

    name = "agrapa"

    def __init__(self, lam0: float = 0.1, lam_max: float = LAMBDA_CAP):
        if not 0.0 <= lam_max <= LAMBDA_CAP:
            raise ValueError(f"lam_max must be in [0, {LAMBDA_CAP}], got {lam_max}")
        if not 0.0 < lam0 <= lam_max:
            raise ValueError(f"lam0 must be in (0, lam_max], got {lam0}")
        self.lam0 = float(lam0)
        self.lam_max = float(lam_max)

    @property
    def n_components(self) -> int:
        return 1

    def init_state(self) -> dict:
        return {"k": 0, "sum_z": 0.0, "sum_q": 0.0}

    def fractions(self, state: dict, eps: float) -> list[float]:
        k = state["k"]
        if k == 0:
            return [self.lam0]
        zbar = state["sum_z"] / k
        qbar = state["sum_q"] / k
        mu = eps - zbar
        var = qbar - zbar * zbar
        den = var + mu * mu
        if den <= 0.0 or mu <= 0.0:
            return [0.0]
        return [min(mu / den, self.lam_max)]

    def observe(self, state: dict, z: float, q: float) -> None:
        state["k"] += 1
        state["sum_z"] += z
        state["sum_q"] += q

    def config(self) -> dict:
        return {"name": self.name, "lam0": self.lam0, "lam_max": self.lam_max}


class FixedMixture(BettingScheme):
    """Uniform mixture over a fixed grid of fractions (default {0.1, 0.2, 0.4})."""

    name = "mixture"

    def __init__(self, lambdas: Sequence[float] = (0.1, 0.2, 0.4)):
        lambdas = tuple(float(x) for x in lambdas)
        if not lambdas:
            raise ValueError("lambdas must be non-empty")
        for lam in lambdas:
            if not 0.0 <= lam <= LAMBDA_CAP:
                raise ValueError(f"each lambda must be in [0, {LAMBDA_CAP}], got {lam}")
        self.lambdas = lambdas

    @property
    def n_components(self) -> int:
        return len(self.lambdas)

    def fractions(self, state: dict, eps: float) -> list[float]:
        return list(self.lambdas)

    def config(self) -> dict:
        return {"name": self.name, "lambdas": list(self.lambdas)}


def make_betting(betting: str | BettingScheme = "agrapa", **kwargs) -> BettingScheme:
    """Build a scheme from a name (``"agrapa"`` or ``"mixture"``) or pass one through."""
    if isinstance(betting, BettingScheme):
        return betting
    key = str(betting).lower()
    if key == "agrapa":
        return AGRAPA(**_pick(kwargs, ("lam0", "lam_max")))
    if key in ("mixture", "fixed", "fixed_mixture"):
        return FixedMixture(**_pick(kwargs, ("lambdas",)))
    raise ValueError(f"unknown betting scheme {betting!r}; use 'agrapa' or 'mixture'")


def betting_from_config(cfg: dict) -> BettingScheme:
    cfg = dict(cfg)
    name = cfg.pop("name")
    return make_betting(name, **cfg)


def _pick(kwargs: dict, keys: Iterable[str]) -> dict:
    return {k: kwargs[k] for k in keys if k in kwargs and kwargs[k] is not None}
