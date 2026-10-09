# Conventions

The detector itself is a few dozen lines. Most of the decisions that change a
number live in how a run is turned into rounds. They are collected here.

## Rounds and steps

A **round** is one paired evaluation: the incumbent theta_{t-1} and a candidate
theta'_t scored on the same fixed validation items. A **step** is one iteration of
the host system's loop.

* `Monitor.update` is called once per round. A step that produces no candidate
  (SkillOpt `skip_no_patches` / `skip_no_rewrite`) is not a round.
* `T_alarm` and `nu_hat` are round indices. `alarm_step` maps the alarm back to
  the host's step; costs are always counted through `alarm_step`, including the
  skipped steps before it, because those steps still spent tokens.
* In the paper's runs the two indices only differ for SkillOpt / GPT-5.6 Luna /
  SearchQA (steps 4 and 11 were skipped, so round 10 is step 12) and in a few
  late cells of Table 7 (LiveMath, round 12 is step 15).

## Paired outcomes

* `X = c - b` per item, with b and c binary correctness of the incumbent and the
  candidate. `n_up` counts X = +1, `n_down` counts X = -1, all other items are ties.
* Items are matched by id. Only common ids count.
* **Missing pairs.** When SkillOpt answers a candidate from its score cache it
  writes no per-item outcomes for it. The reader then records the round as `n`
  ties and flags it `missing_pairs`. In practice this happens when the candidate
  equals a skill already in the cache, so ties are the honest reading, but the
  flag lets you check.

## Which artifact is returned

`Monitor.selected` is the `incumbent` passed at round `nu_hat`, that is the
incumbent that round `nu_hat` was compared against. This is theta_{nu_hat - 1}
in the paper: the last retained artifact before expected improvement fell below
eps.

* SkillOpt: the trace stores `incumbent` (the last accepted step, for example
  `step_0003`) and `incumbent_file` (`skills/skill_vXXXX.md`, the live skill
  SkillOpt saved after the previous step). They differ when a slow update
  rewrote the skill after the last accept; `incumbent_file` is then the exact
  theta_{t-1}.
* GEPA: round t is candidate t and its incumbent is its parent, so
  `selected` is the parent of candidate `nu_hat`.

## SkillOpt slow updates

SkillOpt (v0.1.0, default `slow_update_gate_with_selection: false`) injects
epoch-level guidance into the live skill at the end of every epoch from epoch 2
on, without re-evaluating it. Its per-item cache, and therefore the baseline in
`paired_outcomes.json`, keeps the outcomes of the last step-level accept. Rounds
after such an injection compare the candidate against a stale baseline.

* The reader flags these rounds `baseline_stale` (or `baseline_possibly_stale`
  when the history cannot tell whether the update produced content).
* `stops detect` prints the flags raised up to the alarm.
* The trainer patch fixes it going forward: with `stops.refresh_baseline: true`
  (default) the incumbent is evaluated once on the selection set after it
  changes, at the cost of one extra selection pass per slow update.

## GEPA

* GEPA accepts on a minibatch and only then scores the candidate on the full
  validation set. Rejected proposals have no full-validation scores, so rounds
  are accepted candidates only.
* Baseline = the candidate's parent (GEPA samples parents from its Pareto front),
  not candidate t-1.
* Tokens are estimated as metric calls times a tokens-per-call constant supplied
  by the user (the paper uses the matched SkillOpt rollout: 2249 for SearchQA,
  1354 for GSM8K). The alarm and change point do not depend on this.

## Betting

* aGRAPA: lambda_0 = 0.1 on a gambler's entry round, then the plug-in from the
  gambler's own window, capped at 1/2.
* Fixed mixture: Lambda = {0.1, 0.2, 0.4}, uniform weights.
* Ties in argmax_s W^(s) go to the earliest start.
* Wealth is kept in log space; no clipping.
