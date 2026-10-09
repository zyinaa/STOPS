"""Reproduce the paper's tables from the committed traces (no API calls).

    python paper/reproduce.py              # print tables, write paper/output/tables.md
    python paper/reproduce.py --figures    # also draw trajectory figures (needs matplotlib)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stops import Trace, load_outcomes, paired_comparison
from stops.evaluate import tokens_per_point

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "output"


def _fmt_pair(r):
    return "-" if r.t_alarm is None else f"{r.t_alarm}/{r.nu_hat}"


def cell_rows(cells, eps, delta):
    rows = []
    for c in cells:
        tr = Trace.load(ROOT / "traces" / f"{c['id']}.json")
        ra = tr.replay(eps=eps, delta=delta, betting="agrapa")
        rm = tr.replay(eps=eps, delta=delta, betting="mixture")
        row = {
            "id": c["id"], "system": c["system"], "benchmark": c["benchmark"], "model": c["model"],
            "n": tr.meta["n_items"], "N": tr.meta["budget"],
            "agrapa": _fmt_pair(ra), "mixture": _fmt_pair(rm), "alarm_step": ra.alarm_step,
            "returned": ra.returned,
            "tok_alarm_M": None if ra.tokens_to_alarm is None else ra.tokens_to_alarm / 1e6,
            "tok_full_M": None if tr.total_tokens() is None else tr.total_tokens() / 1e6,
            "saved_pct": None if ra.tokens_saved is None else 100 * ra.tokens_saved,
            "flags": ",".join(ra.flags_through_alarm()),
        }
        test = c.get("test", {})
        ret_key = test.get("returned") or test.get("evaluated_as_returned")
        if ret_key and test.get("full"):
            res = paired_comparison(
                load_outcomes(ROOT / "traces" / "test" / f"{ret_key}.json"),
                load_outcomes(ROOT / "traces" / "test" / f"{test['full']}.json"),
            )
            base = test.get("baseline_acc")
            row.update(
                test_artifact=ret_key.split("__")[-1] + ("" if test.get("returned") else " (*)"),
                acc_base=base, acc_full=100 * res["acc_full"], acc_ret=100 * res["acc_returned"],
                delta_pp=100 * res["delta"], ci_pp=(100 * res["ci"][0], 100 * res["ci"][1]),
            )
            if base is not None and row["tok_full_M"]:
                fp = tokens_per_point(row["tok_full_M"], res["acc_full"], base / 100)
                rp = tokens_per_point(row["tok_alarm_M"], res["acc_returned"], base / 100)
                row.update(tok_per_pt_full=fp, tok_per_pt_ret=rp, eff=(fp / rp) if fp and rp else None)
        rows.append(row)
    return rows


def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def render(rows, t7, eps, delta, t7_eps, t7_delta) -> str:
    L = [f"# Reproduced results (eps={eps}, delta={delta})", ""]
    L.append("## Stopping and cost (Tables 1-6)")
    L.append("| cell | n | N | aGRAPA T/nu | mix T/nu | alarm step | returned | tok@alarm M | full M | saved % | data flags |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r['id']} | {r['n']} | {r['N']} | {r['agrapa']} | {r['mixture']} | {r['alarm_step'] or '-'} | "
            f"{r['returned'] or '-'} | {_f(r['tok_alarm_M'])} | {_f(r['tok_full_M'])} | {_f(r['saved_pct'], 1)} | {r['flags'] or ''} |"
        )
    L += ["", "## Unseen test (Appendix C)", "(*) the file evaluated is not the returned artifact; see paper/cells.json", ""]
    L.append("| cell | test artifact | baseline % | full % | returned % | Delta pp | 95% CI pp | tok/pt full | tok/pt ret | gain x |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "acc_ret" not in r:
            continue
        L.append(
            f"| {r['id']} | {r['test_artifact']} | {_f(r['acc_base'])} | {_f(r['acc_full'])} | {_f(r['acc_ret'])} | "
            f"{r['delta_pp']:+.2f} | [{r['ci_pp'][0]:+.2f}, {r['ci_pp'][1]:+.2f}] | {_f(r.get('tok_per_pt_full'))} | "
            f"{_f(r.get('tok_per_pt_ret'))} | {_f(r.get('eff'), 1)} |"
        )
    L += ["", "## Sensitivity to (eps, delta), aGRAPA (Table 7)", ""]
    for cid, grid in t7.items():
        L.append(f"**{cid}**  (rows eps, columns delta = {t7_delta})")
        L.append("| eps | " + " | ".join(str(d) for d in t7_delta) + " |")
        L.append("|---" * (len(t7_delta) + 1) + "|")
        for e, row in zip(t7_eps, grid):
            L.append(f"| {e} | " + " | ".join(row) + " |")
        L.append("")
    return "\n".join(L)


def table7(ids, eps_grid, delta_grid):
    out = {}
    for cid in ids:
        tr = Trace.load(ROOT / "traces" / f"{cid}.json")
        out[cid] = [[_fmt_pair(tr.replay(eps=e, delta=d)) for d in delta_grid] for e in eps_grid]
    return out


def figures(cells, eps, delta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    for c in cells:
        tr = Trace.load(ROOT / "traces" / f"{c['id']}.json")
        r = tr.replay(eps=eps, delta=delta)
        xs = [s["step"] for s in tr.steps if s.get("incumbent_score") is not None]
        ys = [s["incumbent_score"] for s in tr.steps if s.get("incumbent_score") is not None]
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(xs, ys, marker="o", ms=3, lw=1.5, label="incumbent validation accuracy")
        if r.alarm_step is not None:
            ax.axvline(r.alarm_step, color="#c0392b", ls="--", lw=1.2, label=f"stop (step {r.alarm_step})")
            nu_step = tr.rounds()[r.nu_hat - 1]["step"]
            ax.axvline(nu_step, color="#7d3c98", ls=":", lw=1.2, label=f"nu_hat (step {nu_step})")
        ax.set_xlabel("evolution step")
        ax.set_ylabel("validation accuracy")
        ax.set_title(f"{c['benchmark']} / {c['system']} / {c['model']}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / f"{c['id']}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()

    spec = json.loads((ROOT / "paper" / "cells.json").read_text(encoding="utf-8"))
    cells = spec["cells"]
    t7_eps, t7_delta = spec["sensitivity_T7"]["eps"], spec["sensitivity_T7"]["delta"]
    rows = cell_rows(cells, a.eps, a.delta)
    t7 = table7(spec["sensitivity_T7"]["grids"], t7_eps, t7_delta)
    md = render(rows, t7, a.eps, a.delta, t7_eps, t7_delta)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    if a.figures:
        figures(cells, a.eps, a.delta)
        print(f"\nfigures written to {OUT}")


if __name__ == "__main__":
    main()
