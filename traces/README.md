# Traces

`<cell>.json` holds one run per paper cell in the `stops.trace/v1` format (see
`src/stops/trace.py`): per-step paired counts, gate decisions, incumbent labels,
token counts and data-quality flags. They were produced with
`scripts/build_traces.py` from the authors' SkillOpt and GEPA run directories.
Everything in `paper/reproduce.py` and `tests/test_regression.py` runs from these
files alone.

`test/<cell>__<artifact>.json` holds per-item unseen-test correctness
(`{item_id: 0/1}`) for the artifacts compared in the paper.

| file | source (authors' working directories) |
|---|---|
| skillopt_searchqa_deepseek__step_0001 | SkillOpt/outputs/cand_eval_deepseek/step1/results.jsonl |
| skillopt_searchqa_deepseek__slow_update_epoch_04 | SkillOpt/outputs/cand_eval_deepseek/slow_ep4/results.jsonl |
| skillopt_gsm8k_deepseek__step_0001 | SkillOpt/outputs/cand_eval_gsm8k_deepseek/detector_stop_step1/results.jsonl |
| skillopt_gsm8k_deepseek__full_run_best | SkillOpt/outputs/cand_eval_gsm8k_deepseek/full_run_best/results.jsonl |
| skillopt_officeqa_deepseek__initial | SkillOpt/outputs/repro_baseline_test/results.jsonl |
| skillopt_officeqa_deepseek__step_0003 | SkillOpt/outputs/repro_step3_test/results.jsonl |
| skillopt_officeqa_deepseek__slow_update_epoch_07 | SkillOpt/outputs/repro_epoch07_test/results.jsonl |
| skillopt_officeqa_deepseek__best_skill | SkillOpt/outputs/repro_best_test/results.jsonl |
| skillopt_searchqa_qwen__step_0008 | SkillOpt/outputs/return_tests/searchqa_rqa_v2_step8/results.jsonl |
| skillopt_searchqa_qwen__best_skill | SkillOpt/outputs/searchqa_rqa_v2/test_eval/results.jsonl |
| skillopt_searchqa_luna__step_0008 | SkillOpt/outputs/return_tests/searchqa_luna_high3_step8/results.jsonl |
| skillopt_searchqa_luna__best_skill | SkillOpt/outputs/searchqa_luna_high3/test_eval/results.jsonl |
| skillopt_gsm8k_qwen__step_0005 | SkillOpt/outputs/return_tests/gsm8k_qwen_step5/results.jsonl |
| skillopt_gsm8k_qwen__best_skill | SkillOpt/outputs/gsm8k_qwen/test_eval/results.jsonl |
| skillopt_gsm8k_luna__step_0001 | SkillOpt/outputs/return_tests/gsm8k_luna_high_step1/results.jsonl |
| skillopt_gsm8k_luna__best_skill | SkillOpt/outputs/gsm8k_luna_high/test_eval/results.jsonl |
| skillopt_searchqa_deepseek_val100__step_0003 | SkillOpt/outputs/val100_tests/searchqa_deepseek_val100_return_step3/results.jsonl |
| skillopt_searchqa_deepseek_val100__best_skill | SkillOpt/outputs/val100_tests/searchqa_deepseek_val100_best/results.jsonl |
| skillopt_gsm8k_deepseek_val100__step_0001 | SkillOpt/outputs/val100_tests/gsm8k_deepseek_val100_return_step1/results.jsonl |
| skillopt_gsm8k_deepseek_val100__best_skill | SkillOpt/outputs/gsm8k_deepseek_val100/test_eval/results.jsonl |
| gepa_searchqa_deepseek__cand_1 | seas-cross-system/runs/gepa_sq_peritem.json, key return_cand1 |
| gepa_searchqa_deepseek__cand_10 | seas-cross-system/runs/gepa_sq_peritem.json, key best_cand10 |

Item ids are the benchmark ids used by SkillOpt (GEPA files use the test-set
index). No prompts, answers or model outputs are stored.
