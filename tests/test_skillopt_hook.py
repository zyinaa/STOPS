import json

from stops.integrations.skillopt import SkillOptHook, per_item_counts


def items(bits):
    return {f"i{k}": {"hard": float(b)} for k, b in enumerate(bits)}


def test_per_item_counts():
    assert per_item_counts(items([0, 1, 1]), items([1, 1, 0])) == (1, 1, 3, False)
    assert per_item_counts(items([0, 1, 1]), {}) == (0, 0, 3, True)


def run_plateau(hook, start=1, stop=None):
    """Two useful rounds then a plateau, as SkillOpt would report them."""
    def solved(level):  # incumbent after `level` useful steps
        return [1 if k < 40 * level else 0 for k in range(200)]

    for step in range(start, (stop or 60) + 1):
        base = solved(min(step - 1, 2))
        if step <= 2:
            cand = solved(step)
        else:
            cand = list(base)
            cand[step % 200] = 1 - cand[step % 200]  # one flip, no real gain
        hook.on_round(step, items(base), items(cand), incumbent_skill=f"skill text {step - 1}")
        if hook.should_stop:
            return step
    return None


def test_hook_stops_and_selects(tmp_path):
    hook = SkillOptHook({"stops": {"enabled": True}}, tmp_path)
    stop_step = run_plateau(hook)
    assert stop_step is not None
    res = hook.finalize()
    assert res["alarm"] and res["alarm_step"] == stop_step
    assert res["nu_hat"] == 3
    assert hook.selected_skill() == "skill text 2"  # theta_{nu_hat-1}
    assert (tmp_path / "stops" / "result.json").exists()
    lines = (tmp_path / "stops" / "rounds.jsonl").read_text().splitlines()
    assert len(lines) == stop_step and json.loads(lines[-1])["alarm"]


def test_hook_log_only_mode_never_stops(tmp_path):
    hook = SkillOptHook({"stops": {"enabled": True, "stop": False}}, tmp_path)
    assert run_plateau(hook, stop=30) is None
    assert hook.monitor.alarm and not hook.should_stop


def test_hook_resumes_from_saved_state(tmp_path):
    hook = SkillOptHook({"stops": {"enabled": True}}, tmp_path)
    run_plateau(hook, stop=4)
    # simulate a crash + resume: new hook, steps 3-4 replayed by the trainer
    resumed = SkillOptHook({"stops": {"enabled": True}}, tmp_path)
    assert resumed.monitor.t == 4
    stop_resumed = run_plateau(resumed, start=3)
    fresh = SkillOptHook({"stops": {"enabled": True}}, tmp_path / "fresh")
    assert stop_resumed == run_plateau(fresh)
    assert resumed.monitor.nu_hat == fresh.monitor.nu_hat
