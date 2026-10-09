"""Read a GEPA run directory into a :class:`~stops.trace.Trace`.

Uses ``gepa_state.bin`` (a pickled dict written by ``gepa.optimize``). Only plain
Python containers are expected inside; loading refuses pickles that reference any
class or function.

Mapping onto the paper's rounds (see docs/conventions.md):

* GEPA accepts a proposal on a minibatch and only then scores it on the full
  validation set. Rejected proposals never get full validation scores, so the
  detector sees accepted candidates only: round t is GEPA candidate t.
* The baseline of round t is the candidate's parent (GEPA's Pareto-sampled
  incumbent), not necessarily candidate t-1.
* GEPA logs metric calls, not tokens. Tokens are estimated as metric calls times
  a tokens-per-call constant you supply (the paper uses the matched SkillOpt
  rollout: 2249 on SearchQA, 1354 on GSM8K).
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

from ..trace import Trace


class _PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):  # noqa: D401
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from gepa_state.bin")


def load_gepa_state(run_dir: str | Path) -> dict:
    raw = Path(run_dir, "gepa_state.bin").read_bytes()
    return _PlainUnpickler(io.BytesIO(raw)).load()


def read_gepa(
    run_dir: str | Path,
    *,
    tokens_per_call: float | None = None,
    threshold: float = 0.5,
    name: str | None = None,
    benchmark: str | None = None,
    model: str | None = None,
) -> Trace:
    run_dir = Path(run_dir)
    st = load_gepa_state(run_dir)
    scores = st["prog_candidate_val_subscores"]
    parents = st["parent_program_for_candidate"]
    calls = st.get("num_metric_calls_by_discovery")

    def b(x) -> int:
        return 1 if float(x) >= threshold else 0

    steps = []
    n_items = 0
    for t in range(1, len(scores)):
        par = parents[t][0]
        cur, base = scores[t], scores[par]
        common = cur.keys() & base.keys()
        up = sum(1 for i in common if b(cur[i]) > b(base[i]))
        down = sum(1 for i in common if b(cur[i]) < b(base[i]))
        n_items = max(n_items, len(common))
        flags = [] if len(parents[t]) == 1 else ["merge_candidate"]
        cand_score = sum(float(v) for v in cur.values()) / max(len(cur), 1)
        best_so_far = max(cand_score, steps[-1]["incumbent_score"] if steps else
                          sum(float(v) for v in scores[0].values()) / max(len(scores[0]), 1))
        tok = None
        if calls is not None and tokens_per_call is not None:
            tok = round((calls[t] - calls[t - 1]) * tokens_per_call)
        steps.append(
            {
                "step": t,
                "evaluated": True,
                "n_up": up,
                "n_down": down,
                "n": len(common),
                "accepted": True,
                "incumbent": f"cand_{par}",
                "candidate": f"cand_{t}",
                "candidate_score": cand_score,
                "incumbent_score": best_so_far,  # best full-validation score so far
                "metric_calls": None if calls is None else int(calls[t]),
                "tokens": tok,
                "flags": flags,
            }
        )

    full_tokens = None
    if calls is not None and tokens_per_call is not None:
        full_tokens = round(calls[-1] * tokens_per_call)
    meta = {
        "name": name or run_dir.name,
        "system": "gepa",
        "run": run_dir.name,
        "benchmark": benchmark,
        "model": model,
        "n_items": n_items,
        "budget": len(scores) - 1,
        "tokens_per_call": tokens_per_call,
        "token_note": "estimated as metric calls x tokens_per_call",
        "round_semantics": "round t = accepted candidate t; baseline = its parent",
    }
    full_run = {
        "artifact": "GEPA best candidate (result.best_idx)",
        "metric_calls": None if calls is None else int(calls[-1]),
        "tokens": full_tokens,
    }
    return Trace(meta=meta, steps=steps, full_run=full_run)
