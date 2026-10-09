"""The paper's numbers, recomputed from the committed traces."""

import json
from pathlib import Path

import pytest

from stops import Trace, load_outcomes, paired_comparison

ROOT = Path(__file__).resolve().parents[1]
PAPER = json.loads((ROOT / "paper" / "cells.json").read_text(encoding="utf-8"))
CELLS = {c["id"]: c for c in PAPER["cells"]}


def trace(cid):
    return Trace.load(ROOT / "traces" / f"{cid}.json")


@pytest.mark.parametrize("cid", list(CELLS))
@pytest.mark.parametrize("betting", ["agrapa", "mixture"])
def test_alarm_and_change_point(cid, betting):
    exp = CELLS[cid]["expected"].get(betting)
    if exp is None:
        pytest.skip("not reported")
    r = trace(cid).replay(betting=betting)
    assert [r.t_alarm, r.nu_hat] == exp


@pytest.mark.parametrize("cid", list(CELLS))
def test_returned_artifact_and_tokens(cid):
    exp = CELLS[cid]["expected"]
    r = trace(cid).replay()
    if "returned" in exp:
        assert r.returned == exp["returned"]
    if "alarm_step" in exp:
        assert r.alarm_step == exp["alarm_step"]
    if "tokens_to_alarm_M" in exp:
        assert round(r.tokens_to_alarm / 1e6, 2) == exp["tokens_to_alarm_M"]
        assert round(r.trace.total_tokens() / 1e6, 2) == exp["tokens_full_M"]
        assert round(100 * r.tokens_saved, 1) == exp["tokens_saved_pct"]
    if "flags_through_alarm" in exp:
        assert sorted(r.flags_through_alarm()) == sorted(exp["flags_through_alarm"])


@pytest.mark.parametrize("cid", list(PAPER["sensitivity_T7"]["grids"]))
def test_sensitivity_grid_table7(cid):
    s = PAPER["sensitivity_T7"]
    tr = trace(cid)
    for i, eps in enumerate(s["eps"]):
        for j, delta in enumerate(s["delta"]):
            r = tr.replay(eps=eps, delta=delta)
            got = "-" if r.t_alarm is None else f"{r.t_alarm}/{r.nu_hat}"
            assert got == s["grids"][cid][i][j], (eps, delta)


def test_returned_rule_matches_trace_for_every_cell():
    """theta_{nu_hat-1} is the incumbent recorded at round nu_hat."""
    for cid in CELLS:
        tr = trace(cid)
        r = tr.replay()
        if r.nu_hat is None:
            continue
        assert r.returned == tr.rounds()[r.nu_hat - 1]["incumbent"]


TEST_CASES = [
    (cid, c["test"]) for cid, c in CELLS.items()
    if "test" in c and "full" in c["test"]
]


@pytest.mark.parametrize("cid,test", TEST_CASES, ids=[c for c, _ in TEST_CASES])
def test_unseen_test_files_load_and_pair(cid, test):
    ret_key = test.get("returned") or test.get("evaluated_as_returned")
    ret = load_outcomes(ROOT / "traces" / "test" / f"{ret_key}.json")
    full = load_outcomes(ROOT / "traces" / "test" / f"{test['full']}.json")
    res = paired_comparison(ret, full)
    assert res["N"] == len(ret) == len(full)
    lo, hi = res["ci"]
    assert lo <= res["delta"] <= hi
    if "returned" in test:
        # the file must be the artifact the stopping rule returns
        assert test["returned"].endswith(CELLS[cid]["expected"]["returned"])
