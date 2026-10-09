"""Command line: ``stops extract | detect | watch | compare``."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from .evaluate import load_outcomes, paired_comparison
from .monitor import Monitor
from .readers import read_gepa, read_skillopt
from .trace import Trace


def _add_detector_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eps", type=float, default=0.01, help="worthwhile-gain threshold (default 0.01)")
    p.add_argument("--delta", type=float, default=0.05, help="false-alarm control (default 0.05)")
    p.add_argument("--betting", choices=["agrapa", "mixture"], default="agrapa")


def _read(system: str, run_dir: str, tokens_per_call: float | None) -> Trace:
    if system == "skillopt":
        return read_skillopt(run_dir)
    if system == "gepa":
        return read_gepa(run_dir, tokens_per_call=tokens_per_call)
    raise SystemExit(f"unknown system {system!r}")


def cmd_extract(a) -> int:
    tr = _read(a.system, a.run_dir, a.tokens_per_call)
    if a.name:
        tr.meta["name"] = a.name
    for k in ("benchmark", "model"):
        if getattr(a, k):
            tr.meta[k] = getattr(a, k)
    tr.save(a.output)
    print(f"wrote {a.output}: {len(tr.rounds())} rounds / {len(tr.steps)} steps, n_items={tr.meta['n_items']}")
    return 0


def _print_rounds(mon: Monitor) -> None:
    print(f"{'t':>3} {'step':>5} {'n_up':>5} {'n_dn':>5} {'n':>5} {'Z_t':>8} {'M_t':>12} {'s*':>4}")
    for r in mon.history:
        m = math.exp(r.log_m) if r.log_m < 700 else float("inf")
        flag = "  <== alarm" if r.alarm else ""
        print(f"{r.t:>3} {str(r.step):>5} {r.n_up:>5} {r.n_down:>5} {r.n:>5} {r.z:>+8.4f} {m:>12.4g} {r.s_star:>4}{flag}")


def cmd_detect(a) -> int:
    tr = Trace.load(a.trace)
    res = tr.replay(eps=a.eps, delta=a.delta, betting=a.betting)
    if a.json:
        print(json.dumps(res.as_dict(), indent=2))
        return 0
    print(f"{tr.name}: system={tr.meta.get('system')} n_items={tr.meta.get('n_items')} budget={tr.meta.get('budget')}")
    _print_rounds(res.monitor)
    print(res.monitor.summary())
    if res.t_alarm:
        print(f"return: {res.returned}  (theta_(nu_hat-1), incumbent at round {res.nu_hat})")
        if res.returned_file and res.returned_file != res.returned:
            print(f"        file: {res.returned_file}")
        if res.tokens_saved is not None:
            print(f"tokens: {res.tokens_to_alarm:,} of {tr.total_tokens():,}  saved {100 * res.tokens_saved:.1f}%")
        flags = res.flags_through_alarm()
        if flags:
            print(f"data flags up to the alarm: {', '.join(flags)}")
    return 0


def cmd_watch(a) -> int:
    """Poll a running run directory and feed new rounds to one monitor."""
    state_path = Path(a.state) if a.state else Path(a.run_dir) / "stops_monitor.json"
    mon = Monitor(eps=a.eps, delta=a.delta, betting=a.betting)
    fed = 0
    print(f"watching {a.run_dir} every {a.interval}s ({mon.summary()})", flush=True)
    while True:
        try:
            tr = _read(a.system, a.run_dir, a.tokens_per_call)
            rounds = tr.rounds()
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            rounds = []
        for s in rounds[fed:]:
            mon.update_counts(s["n_up"], s["n_down"], s["n"], incumbent=s.get("incumbent"), step=s["step"])
            r = mon.history[-1]
            print(f"round {r.t} (step {r.step}): Z={r.z:+.4f} M={math.exp(min(r.log_m, 700)):.4g} s*={r.s_star}", flush=True)
            fed += 1
            if mon.alarm:
                break
        mon.save(state_path)
        if mon.alarm:
            print(f"ALARM: {mon.summary()}\nreturn: {mon.selected}", flush=True)
            if a.stop_file:
                Path(a.stop_file).write_text(json.dumps({"alarm_step": mon.alarm_step, "selected": mon.selected}))
                print(f"wrote stop file {a.stop_file}", flush=True)
            return 0
        if a.once:
            return 0
        time.sleep(a.interval)


def cmd_compare(a) -> int:
    r = paired_comparison(load_outcomes(a.returned), load_outcomes(a.full))
    lo, hi = r["ci"]
    print(
        f"N={r['N']}  returned={100 * r['acc_returned']:.2f}  full={100 * r['acc_full']:.2f}  "
        f"Delta={100 * r['delta']:+.2f} pp  95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]  "
        f"(n10={r['n10']}, n01={r['n01']})"
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stops", description="When to stop a self-evolving loop, and what to return.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="read a run directory into a trace JSON")
    p.add_argument("system", choices=["skillopt", "gepa"])
    p.add_argument("run_dir")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--tokens-per-call", type=float, default=None, help="GEPA only: token estimate per metric call")
    p.add_argument("--name")
    p.add_argument("--benchmark")
    p.add_argument("--model")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("detect", help="replay the detector over a trace")
    p.add_argument("trace")
    _add_detector_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("watch", help="monitor a running run directory")
    p.add_argument("system", choices=["skillopt", "gepa"])
    p.add_argument("run_dir")
    _add_detector_args(p)
    p.add_argument("--interval", type=float, default=60.0, help="seconds between polls")
    p.add_argument("--stop-file", help="write this file on alarm (GEPA's FileStopper can watch it)")
    p.add_argument("--state", help="where to save the monitor state (default RUN_DIR/stops_monitor.json)")
    p.add_argument("--tokens-per-call", type=float, default=None)
    p.add_argument("--once", action="store_true", help="poll once and exit")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("compare", help="paired unseen-test comparison of two artifacts")
    p.add_argument("returned", help="per-item results of the returned artifact (.jsonl or {id: 0/1} .json)")
    p.add_argument("full", help="per-item results of the full-run artifact")
    p.set_defaults(func=cmd_compare)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
