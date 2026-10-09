"""Plug the monitor into a self-evolving loop. Runs in a second, no API needed.

The "system" below is a toy: an artifact has a skill level, each validation item
has a difficulty, and an item is solved when skill plus a small item-level
jitter beats its difficulty. The first three proposals add real skill; after
that, proposals only reshuffle which borderline items get solved. Swap
`propose`, `evaluate` and `gate` for your own system; the three marked lines
stay the same.

    python examples/minimal_loop.py
"""

import random

from stops import Monitor

N_VAL, BUDGET = 200, 40
rng = random.Random(0)
difficulty = [rng.random() for _ in range(N_VAL)]


def evaluate(theta):
    """Per-item 0/1 correctness of artifact `theta` on the fixed validation set."""
    skill, seed = theta
    jitter = random.Random(seed)
    return [int(skill + jitter.gauss(0, 0.05) > d) for d in difficulty]


def propose(theta, t):
    skill, _ = theta
    gain = 0.08 if t <= 3 else rng.gauss(0.0, 0.003)   # real gains, then none
    return (skill + gain, rng.randrange(10**9))


def gate(theta, cand, b, c):
    return cand if sum(c) > sum(b) else theta


theta = (0.4, 0)
names = {theta: "theta_0"}
mon = Monitor(eps=0.01, delta=0.05)                       # (1) create once

for t in range(1, BUDGET + 1):
    cand = propose(theta, t)
    b, c = evaluate(theta), evaluate(cand)
    mon.update(b, c, incumbent=names[theta], step=t)      # (2) feed every round
    print(f"round {t:2d}: val acc {sum(b) / N_VAL:.3f} -> {sum(c) / N_VAL:.3f}   M_t={mon.wealth:8.3g}  (alarm at {mon.threshold:g})")
    if mon.alarm:                                         # (3) stop and return
        print(f"\nstop at round {mon.t_alarm} of {BUDGET}; change point nu_hat={mon.nu_hat}")
        print(f"return {mon.selected}: the incumbent when worthwhile improvement ended")
        break
    theta = gate(theta, cand, b, c)
    names.setdefault(theta, f"theta_{t}")
else:
    print("no alarm: improvement never stalled within the budget")
