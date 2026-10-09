# STOPS: Sequential Tests for Online stopping and Plateau-based Selection

**When to stop a self-evolving LLM loop, and which artifact to return.**

Reference implementation of *When Is Enough Enough in Self-Evolving LLM Systems?*
(Enoch Yin, Bin Liu, Zhengling Qi).

Self-evolving systems (SkillOpt, GEPA, and others) repeatedly propose an update to
a prompt or skill, score it against the current one on a validation set, and keep
it if it wins. They usually run for a fixed budget. `stops` watches the per-item
paired outcomes that loop already produces and answers two questions online:

* **When to stop.** A restart e-detector built from betting martingales raises an
  alarm once there is evidence that the expected gain of a proposal has fallen
  below a worthwhile threshold `eps`. Under continued worthwhile improvement the
  average run length to a false alarm is at least `1/delta` (Ville's inequality).
* **What to return.** The restart with the most wealth at the alarm estimates when
  worthwhile improvement ended (`nu_hat`); `stops` returns the incumbent from just
  before it, `theta_{nu_hat - 1}`.

It needs no change to the underlying algorithm and no extra model calls.

<p align="center">
  <img src="assets/pipeline.png" alt="STOPS pipeline: a self-evolving loop feeds paired outcomes to the plug-and-play module, which decides when to stop and which artifact to return" width="100%">
</p>

## Install

```bash
git clone <this repo> && cd stops
pip install -e .            # no dependencies; Python >= 3.9
pip install -e ".[dev,plot]"  # tests and figures
```

## Use it in your own loop

```python
from stops import Monitor

mon = Monitor(eps=0.01, delta=0.05)                 # aGRAPA betting by default
for t in range(1, budget + 1):
    cand = propose(inc)
    b, c = evaluate(inc), evaluate(cand)            # per-item 0/1, same items
    mon.update(b, c, incumbent=inc)                 # lists aligned by position, or {item_id: 0/1}
    if mon.alarm:
        return mon.selected                         # θ_{ν̂-1}
    inc = gate(inc, cand)
```

`python examples/minimal_loop.py` runs this against a toy system in a second.
`Monitor.save` / `Monitor.load` persist the full state for long or resumable runs.

## Use it with SkillOpt or GEPA

| | after a run | during a run |
|---|---|---|
| **SkillOpt** | `stops extract skillopt <run_dir> -o run.json` then `stops detect run.json` | apply `integrations/skillopt/skillopt-v0.1.0.patch` and set `stops.enabled: true`; or `stops watch skillopt <run_dir>` |
| **GEPA** | `stops extract gepa <run_dir> --tokens-per-call N -o run.json` then `stops detect run.json` | `stops watch gepa <run_dir> --stop-file <run_dir>/gepa.stop` |

Details: [`integrations/skillopt`](integrations/skillopt/README.md),
[`integrations/gepa`](integrations/gepa/README.md). Reading SkillOpt runs needs
the per-item `paired_outcomes.json` files that the patch writes.

## Choosing eps and delta

* `eps` is the smallest expected per-round validation gain worth another round,
  in accuracy units (0.01 = one point). Larger `eps` stops earlier.
* `delta` sets the evidence bar: the alarm fires at `M_t >= 1/delta`. Smaller
  `delta` stops later and raises fewer false alarms.
* `betting="agrapa"` (default) adapts the bet size per restart;
  `betting="mixture"` averages fixed fractions {0.1, 0.2, 0.4}. Both are valid.
* Resolution scales with the validation size `n`: on a flat plateau the alarm
  needs roughly `log(1/delta) / (n * lambda * eps)` rounds.

## Reproduce the paper

The `traces/` directory holds every run used in the paper as compact JSON, so the
stopping and cost results reproduce without any API calls:

```bash
pytest                          # 130 tests, includes every paper cell
python paper/reproduce.py       # tables -> paper/output/tables.md
python paper/reproduce.py --figures
```

| what | from | API needed |
|---|---|---|
| T_alarm, nu_hat, returned artifact (Tables 1-5, 7) | `traces/*.json` | no |
| tokens to alarm, tokens saved | `traces/*.json` | no |
| paired unseen-test Delta and 95% CI (Appendix C) | `traces/test/*.json` | no |
| new evolution runs | SkillOpt / GEPA | yes |
| unseen-test accuracy of a new artifact | SkillOpt / GEPA evaluation | yes |

`paper/cells.json` lists each cell, its source run and the expected numbers.
To rebuild traces from raw run directories:
`python scripts/build_traces.py --skillopt-outputs ... --gepa-runs ...`.

## Repository layout

```
src/stops/
  monitor.py        Monitor: the online detector and output selection
  betting.py        aGRAPA and fixed-mixture betting fractions
  trace.py          run-independent trace format and replay
  readers/          SkillOpt and GEPA run directory -> trace
  evaluate.py       paired unseen-test comparison
  integrations/     SkillOpt trainer hook
  cli.py            stops extract | detect | watch | compare
integrations/       SkillOpt patch, GEPA notes
traces/             paper runs as traces; traces/test: per-item test outcomes
paper/              cells.json, reproduce.py
examples/           minimal_loop.py
tests/              unit, reference-equivalence and paper regression tests
docs/conventions.md how runs become rounds (skips, missing pairs, slow updates)
```

## Citation

```bibtex
@article{yin2026enough,
  title  = {When Is Enough Enough in Self-Evolving {LLM} Systems?},
  author = {Yin, Enoch and Liu, Bin and Qi, Zhengling},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
