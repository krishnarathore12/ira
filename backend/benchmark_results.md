# Benchmark Results: HiMem vs Baseline (Running Summary)

This report details accuracy scores scaled 0-1, evaluated by GPT-4 as a judge.

## Dataset: ira_long

| Category | Baseline Avg Score | HiMem Avg Score | Count |
|----------|--------------------|-----------------|-------|
| Explicit Recall | 0.000 | 0.333 | 3 |
| Correction/Supersession | 0.667 | 0.333 | 3 |
| Temporal Ordering | 0.000 | 0.333 | 3 |
| Uncertainty/Honesty | 0.667 | 0.667 | 3 |
| Multi-Turn Continuity | 0.333 | 0.333 | 3 |
| Sensitive Memory | 0.667 | 0.667 | 3 |
| Entity Nuance | 0.000 | 0.000 | 3 |
| Epistemic Honesty | 0.333 | 0.333 | 3 |
| Multi-User Isolation | 0.000 | 0.000 | 3 |
| Tone/Warmth | 1.000 | 0.667 | 3 |
| **Overall** | **0.367** | **0.367** | **30** |

## Dataset: REALTALK

| Category | Baseline Avg Score | HiMem Avg Score | Count |
|----------|--------------------|-----------------|-------|
| temporal-reasoning | 0.000 | 0.000 | 45 |
| multi-hop | 0.567 | 0.333 | 30 |
| open-domain | 0.400 | 0.300 | 10 |
| **Overall** | **0.247** | **0.153** | **85** |

