import json
import math
import random

import pytest

from stops import Monitor, paired_counts
from stops.betting import AGRAPA, FixedMixture

# ---------------------------------------------------------------------------
# Reference implementation: the detector used for the paper's experiments,
# copied from the authors' detect_all_verify.py (fixed n per round).
# The packaged Monitor must agree with it exactly on (T_alarm, nu_hat).
# ---------------------------------------------------------------------------
LAMS = (0.1, 0.2, 0.4)
LAM0 = 0.1


def ref_mixture(rnds, N, EPS=0.01, DELTA=0.05):
    th = 1 / DELTA
    G = {}
    alarm = nu = None
    for t, (a, b) in enumerate(rnds, 1):
        up, dn, ze = a, b, N - a - b
        G[t] = {l: 0.0 for l in LAMS}
        for s, wd in G.items():
            for l in LAMS:
                wd[l] = max(min(wd[l] + up * math.log(1 + l * (EPS - 1)) + dn * math.log(1 + l * (EPS + 1)) + ze * math.log(1 + l * EPS), 690), -690)
        ev = {s: sum(math.exp(v) for v in wd.values()) / len(LAMS) for s, wd in G.items()}
        ss = max(ev, key=ev.get)
        M = ev[ss]
        if alarm is None and M >= th:
            alarm, nu = t, ss
    return alarm, nu


def ref_agrapa(rnds, N, EPS=0.01, DELTA=0.05):
    th = 1 / DELTA
    G = {}
    alarm = nu = None
    for t, (a, b) in enumerate(rnds, 1):
        up, dn, ze = a, b, N - a - b
        G[t] = [0.0, 0, 0, 0]
        for s, st in G.items():
            c = st[3]
            if c == 0:
                lam = LAM0
            else:
                Zbar = (st[2] / c) / N
                qbar = (st[1] / c) / N
                mu = EPS - Zbar
                s2 = qbar - Zbar * Zbar
                den = s2 + mu * mu
                lam = 0.0 if (den <= 0 or mu <= 0) else min(mu / den, 0.5)
            st[0] = max(min(st[0] + up * math.log(1 + lam * (EPS - 1)) + dn * math.log(1 + lam * (EPS + 1)) + ze * math.log(1 + lam * EPS), 690), -690)
            st[1] += a + b
            st[2] += a - b
            st[3] += 1
        ev = {s: st[0] for s, st in G.items()}
        ss = max(ev, key=ev.get)
        M = math.exp(ev[ss])
        if alarm is None and M >= th:
            alarm, nu = t, ss
    return alarm, nu


def random_trajectory(rng, N, T):
    """Improvement that fades into a plateau at a random round."""
    change = rng.randint(1, T)
    out = []
    for t in range(1, T + 1):
        p_up, p_dn = (0.15, 0.05) if t < change else (0.06, 0.07)
        up = sum(rng.random() < p_up for _ in range(N))
        dn = sum(rng.random() < p_dn for _ in range(N - up))
        out.append((up, dn))
    return out


@pytest.mark.parametrize("seed", range(60))
def test_matches_reference_implementation(seed):
    rng = random.Random(seed)
    N = rng.choice([18, 24, 40, 100, 200])
    rnds = random_trajectory(rng, N, rng.randint(5, 40))
    for eps in (0.005, 0.01, 0.05):
        for delta in (0.01, 0.05, 0.1):
            got_a = Monitor.replay([(a, b, N) for a, b in rnds], eps=eps, delta=delta, betting="agrapa")
            got_m = Monitor.replay([(a, b, N) for a, b in rnds], eps=eps, delta=delta, betting="mixture")
            assert (got_a.t_alarm, got_a.nu_hat) == ref_agrapa(rnds, N, eps, delta)
            assert (got_m.t_alarm, got_m.nu_hat) == ref_mixture(rnds, N, eps, delta)


# ---------------------------------------------------------------------------
def test_paired_counts_sequences_and_mappings():
    assert paired_counts([0, 1, 1, 0], [1, 1, 0, 0]) == (1, 1, 4)
    assert paired_counts({"a": 0, "b": 1, "z": 1}, {"a": 1, "b": 1, "y": 0}) == (1, 0, 2)
    assert paired_counts([True, False], [1.0, 1.0]) == (1, 0, 2)
    with pytest.raises(ValueError):
        paired_counts([0, 1], [1])
    with pytest.raises(ValueError):
        paired_counts([0.5], [1])
    with pytest.raises(ValueError):
        paired_counts({"a": 1}, {"b": 1})
    with pytest.raises(TypeError):
        paired_counts({"a": 1}, [1])


def test_invalid_arguments():
    with pytest.raises(ValueError):
        Monitor(eps=0)
    with pytest.raises(ValueError):
        Monitor(delta=1.0)
    with pytest.raises(ValueError):
        Monitor(betting="kelly")
    with pytest.raises(ValueError):
        FixedMixture((0.1, 0.7))
    with pytest.raises(ValueError):
        AGRAPA(lam0=0.0)
    with pytest.raises(ValueError):
        Monitor().update_counts(5, 6, 10)


def test_silent_while_improving():
    mon = Monitor()
    for _ in range(40):
        mon.update_counts(30, 5, 200)  # Z = 0.125 per round, far above eps
    assert not mon.alarm
    assert mon.selected is None


def test_plateau_alarm_and_selection():
    mon = Monitor()
    for t in range(1, 4):  # three useful rounds
        mon.update_counts(40, 4, 200, incumbent=f"theta_{t - 1}", step=t)
    t = 3
    while not mon.alarm and t < 50:
        t += 1
        mon.update_counts(6, 8, 200, incumbent=f"theta_{t - 1}", step=t)
    assert mon.alarm
    assert mon.nu_hat == 4  # first plateau round
    assert mon.selected == "theta_3"  # theta_{nu_hat - 1}
    assert mon.alarm_step == mon.t_alarm
    assert mon.history[mon.t_alarm - 1].alarm


def test_alarm_is_sticky():
    mon = Monitor()
    while not mon.alarm:
        mon.update_counts(0, 0, 200, incumbent=mon.t)
    t_alarm, nu = mon.t_alarm, mon.nu_hat
    for _ in range(5):
        mon.update_counts(50, 0, 200)
    assert (mon.t_alarm, mon.nu_hat) == (t_alarm, nu)
    assert sum(r.alarm for r in mon.history) == 1


def test_update_from_per_item_outcomes():
    rng = random.Random(0)
    b = [rng.randint(0, 1) for _ in range(100)]
    c = [rng.randint(0, 1) for _ in range(100)]
    m1, m2 = Monitor(), Monitor()
    m1.update(b, c)
    m2.update_counts(*paired_counts(b, c))
    assert m1.history[0].log_m == m2.history[0].log_m


@pytest.mark.parametrize("betting", ["agrapa", "mixture"])
def test_save_load_continues_identically(tmp_path, betting):
    rng = random.Random(1)
    rnds = random_trajectory(rng, 200, 30)
    full = Monitor(betting=betting)
    half = Monitor(betting=betting)
    for i, (a, b) in enumerate(rnds):
        full.update_counts(a, b, 200, incumbent=f"inc{i}", step=i + 1)
        if i < 10:
            half.update_counts(a, b, 200, incumbent=f"inc{i}", step=i + 1)
    path = tmp_path / "state.json"
    half.save(path)
    resumed = Monitor.load(path)
    for i, (a, b) in enumerate(rnds[10:], start=10):
        resumed.update_counts(a, b, 200, incumbent=f"inc{i}", step=i + 1)
    assert [r.log_m for r in resumed.history] == pytest.approx([r.log_m for r in full.history])
    assert (resumed.t_alarm, resumed.nu_hat, resumed.selected) == (full.t_alarm, full.nu_hat, full.selected)
    json.loads(path.read_text())  # valid JSON


def test_false_alarm_rate_under_null_is_controlled():
    """Under E[Z] = eps exactly, a single restart crosses 1/delta with prob <= delta (Ville)."""
    rng = random.Random(7)
    eps, delta, N, T, reps = 0.02, 0.1, 50, 30, 400
    crossings = 0
    for _ in range(reps):
        mon = Monitor(eps=eps, delta=delta, betting="mixture")
        crossed = False
        for _ in range(T):
            # X = +1 w.p. 0.06, -1 w.p. 0.04  -> E[X] = 0.02 = eps
            up = dn = 0
            for _ in range(N):
                u = rng.random()
                if u < 0.06:
                    up += 1
                elif u < 0.10:
                    dn += 1
            mon.update_counts(up, dn, N)
            crossed |= mon.restart_log_wealth()[1] >= -math.log(delta)
        crossings += crossed
    assert crossings / reps <= delta + 0.03
