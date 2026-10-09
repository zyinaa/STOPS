import json
import pickle

import pytest

from stops.readers import read_gepa, read_skillopt
from stops.readers.gepa import load_gepa_state


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _paired(run, step, base, cand):
    _write(
        run / "steps" / f"step_{step:04d}" / "paired_outcomes.json",
        {
            "step": step,
            "baseline_per_item": {k: {"hard": v} for k, v in base.items()},
            "candidate_per_item": {k: {"hard": v} for k, v in cand.items()},
        },
    )


def _hist(step, epoch, action, origin):
    return {"step": step, "epoch": epoch, "action": action, "current_origin": origin,
            "tokens": {"rollout": {"prompt_tokens": 100, "completion_tokens": 10 * step}}}


@pytest.fixture
def skillopt_run(tmp_path):
    run = tmp_path / "run"
    items = {f"i{k}": 0 for k in range(4)}
    _paired(run, 1, items, {"i0": 1, "i1": 1, "i2": 0, "i3": 0})          # accepted
    _paired(run, 2, {"i0": 1, "i1": 1, "i2": 0, "i3": 0}, {"i0": 1, "i1": 0, "i2": 0, "i3": 0})
    # step 3: skipped, no file
    _paired(run, 4, {"i0": 1, "i1": 1, "i2": 0, "i3": 0}, {})              # cache hit, no per-item
    _paired(run, 5, {"i0": 1, "i1": 1, "i2": 0, "i3": 0}, {"i0": 1, "i1": 1, "i2": 1, "i3": 0})
    history = [
        _hist(1, 1, "accept_new_best", "step_0001"),
        _hist(2, 1, "reject", "step_0001"),
        _hist(3, 2, "skip_no_patches", None),
        _hist(4, 2, "reject", "slow_update_placeholder_epoch_01"),
        _hist(5, 3, "reject", "slow_update_epoch_02"),
    ]
    # epochs: 1 -> steps 1-2 (placeholder after), 2 -> steps 3-4 (real update after), 3 -> step 5
    _write(run / "history.json", history)
    _write(run / "config.json", {"env": "toy", "target_model": "m", "use_slow_update": True, "steps_per_epoch": 2})
    return run


def test_read_skillopt(skillopt_run):
    tr = read_skillopt(skillopt_run)
    assert tr.meta["n_items"] == 4 and tr.meta["benchmark"] == "toy"
    steps = {s["step"]: s for s in tr.steps}
    assert (steps[1]["n_up"], steps[1]["n_down"], steps[1]["n"]) == (2, 0, 4)
    assert (steps[2]["n_up"], steps[2]["n_down"]) == (0, 1)
    assert steps[3]["evaluated"] is False and "no_candidate" in steps[3]["flags"]
    assert (steps[4]["n_up"], steps[4]["n_down"], steps[4]["n"]) == (0, 0, 4)
    assert "missing_pairs" in steps[4]["flags"]
    # incumbents: initial before the first accept, then step_0001
    assert steps[1]["incumbent"] == "initial" and steps[2]["incumbent"] == "step_0001"
    assert steps[2]["incumbent_file"] == "skills/skill_v0001.md"
    # the real slow update after epoch 2 (ended at step 4) makes step 5's baseline stale
    assert "baseline_stale" in steps[5]["flags"]
    assert "baseline_stale" not in steps[4]["flags"]
    assert [s["step"] for s in tr.rounds()] == [1, 2, 4, 5]
    assert tr.total_tokens() == sum(100 + 10 * k for k in range(1, 6))
    assert tr.tokens_through(2) == 100 + 10 + 100 + 20


def test_read_gepa(tmp_path):
    run = tmp_path / "gepa_run"
    run.mkdir()
    scores = [
        {0: 0.0, 1: 0.0, 2: 1.0},
        {0: 1.0, 1: 0.0, 2: 1.0},
        {0: 1.0, 1: 1.0, 2: 0.0},
    ]
    state = {
        "prog_candidate_val_subscores": scores,
        "parent_program_for_candidate": [[None], [0], [1]],
        "num_metric_calls_by_discovery": [0, 10, 25],
    }
    (run / "gepa_state.bin").write_bytes(pickle.dumps(state))
    tr = read_gepa(run, tokens_per_call=100)
    r1, r2 = tr.rounds()
    assert (r1["n_up"], r1["n_down"], r1["incumbent"]) == (1, 0, "cand_0")
    assert (r2["n_up"], r2["n_down"], r2["incumbent"]) == (1, 1, "cand_1")
    assert tr.tokens_through(1) == 1000 and tr.total_tokens() == 2500


def test_gepa_loader_refuses_objects(tmp_path):
    import collections

    run = tmp_path / "bad"
    run.mkdir()
    (run / "gepa_state.bin").write_bytes(pickle.dumps(collections.OrderedDict(a=1)))
    with pytest.raises(pickle.UnpicklingError):
        load_gepa_state(run)
