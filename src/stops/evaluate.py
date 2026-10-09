"""Unseen-test comparison and cost summaries (paper Appendix C and Table 1)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

_ID_KEYS = ("id", "task_id", "uid", "question_id", "question")


def load_outcomes(path: str | Path) -> dict[str, int]:
    """Per-item 0/1 correctness from a file.

    Accepts SkillOpt ``results.jsonl`` (one JSON object per line with an id field
    and ``hard``), or a JSON object ``{item_id: 0/1}``.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".jsonl":
        out = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rid = next((r[k] for k in _ID_KEYS if r.get(k) not in (None, "")), None)
            if rid is None:
                raise ValueError(f"{path}: record without an id field")
            out[str(rid)] = int(round(float(r.get("hard", 0))))
        return out
    d = json.loads(text)
    if isinstance(d, Mapping):
        return {str(k): int(round(float(v))) for k, v in d.items()}
    raise ValueError(f"{path}: expected .jsonl or a JSON object of id -> 0/1")


def paired_comparison(returned: Mapping[str, int], full: Mapping[str, int], z: float = 1.96) -> dict:
    """Returned-vs-full paired accuracy difference on common items.

    Delta = Acc_returned - Acc_full = (n10 - n01) / N, where n10 counts items the
    returned artifact gets right and the full-run artifact gets wrong.
    SE = sqrt((q - Delta^2) / N) with q = (n01 + n10) / N.
    """
    keys = sorted(returned.keys() & full.keys())
    n = len(keys)
    if n == 0:
        raise ValueError("no common items")
    n11 = n10 = n01 = n00 = 0
    for k in keys:
        r, f = int(returned[k]), int(full[k])
        if r and f:
            n11 += 1
        elif r and not f:
            n10 += 1
        elif f and not r:
            n01 += 1
        else:
            n00 += 1
    delta = (n10 - n01) / n
    q = (n01 + n10) / n
    se = math.sqrt(max(q - delta * delta, 0.0) / n)
    return {
        "N": n,
        "acc_returned": (n11 + n10) / n,
        "acc_full": (n11 + n01) / n,
        "delta": delta,
        "se": se,
        "ci": (delta - z * se, delta + z * se),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }


def tokens_per_point(tokens: float, acc: float, baseline_acc: float) -> float | None:
    """Tokens per percentage point of accuracy gained over the baseline."""
    gain_pp = (acc - baseline_acc) * 100.0
    if gain_pp <= 0:
        return None
    return tokens / gain_pp
