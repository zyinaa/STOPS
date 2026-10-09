# SkillOpt integration

`skillopt-v0.1.0.patch` changes one file, `skillopt/engine/trainer.py` of
[SkillOpt v0.1.0](https://github.com/microsoft/SkillOpt/releases/tag/v0.1.0):

1. **Instrumentation.** Each step writes `steps/step_XXXX/paired_outcomes.json`
   with the incumbent's and the candidate's per-item validation outcomes. This is
   what the paper's experiments used, and what `stops extract skillopt` reads.
2. **Monitor hook** (only when the config has `stops.enabled: true`). After the
   paired outcomes are written, the trainer calls
   `stops.integrations.skillopt.SkillOptHook.on_round`; after the step is saved it
   stops if the alarm has fired, skipping that epoch's slow and meta updates.
3. **Baseline refresh** (with the hook enabled, `stops.refresh_baseline: true`).
   After a slow update changes the live skill, the incumbent is evaluated once on
   the selection set so the paired baseline really is theta_{t-1}. See
   `docs/conventions.md`.

With the hook disabled the patched trainer behaves like v0.1.0 plus the
instrumentation file.

## Apply

```bash
git clone https://github.com/microsoft/SkillOpt && cd SkillOpt
git checkout v0.1.0
git apply --ignore-whitespace /path/to/stops/integrations/skillopt/skillopt-v0.1.0.patch
pip install -e . && pip install -e /path/to/stops
```

`--ignore-whitespace` is needed because the upstream file mixes CRLF and LF line
endings. Later SkillOpt releases restructured `trainer.py`; the patch targets
v0.1.0 only.

## Configure

Add to your SkillOpt YAML config:

```yaml
stops:
  enabled: true
  eps: 0.01
  delta: 0.05
  betting: agrapa        # or mixture
  stop: true             # false: monitor and log, never stop
  refresh_baseline: true
  replace_best: true     # on alarm, best_skill.md := theta_{nu_hat-1}
```

## Outputs

Under `<out_root>/stops/`:

| file | content |
|---|---|
| `monitor.json` | full monitor state; a resumed run continues from it |
| `rounds.jsonl` | one line per round (Z_t, log M_t, s*, alarm) for live watching |
| `incumbents/round_XXXX_step_YYYY.md` | the incumbent skill each round was compared against |
| `result.json` | alarm round and step, nu_hat, selected artifact |

On alarm with `replace_best: true`, `best_skill.md` (and SkillOpt's own test
evaluation, if enabled) uses the selected artifact.

## Without patching the trainer

If you already have the instrumentation, you can watch a run from another
terminal; it does not stop the run, it tells you when to:

```bash
stops watch skillopt outputs/my_run --interval 60
```

## Status

The hook logic is unit-tested (`tests/test_skillopt_hook.py`). The patched
trainer compiles and the patch applies cleanly to v0.1.0, but an end-to-end
SkillOpt run with the hook enabled has not been executed yet.
