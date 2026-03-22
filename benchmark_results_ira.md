# Benchmark Results: HiMem vs Baseline (Running Summary)

This report details accuracy scores scaled 0-1, evaluated by GPT-4 as a judge.


## Dataset: ira_long

| Category | Baseline Avg Score | HiMem Avg Score |
|----------|--------------------|-----------------|
| Explicit Recall | 0.286 | 0.857 |
| Correction/Supersession | 0.524 | 0.833 |
| Temporal Ordering | 0.238 | 0.810 |
| Uncertainty/Honesty | 0.762 | 0.905 |
| Multi-Turn Continuity | 0.619 | 0.881 |
| Sensitive Memory | 0.571 | 0.924 |
| Entity Nuance | 0.190 | 0.762 |
| Epistemic Honesty | 0.619 | 0.857 |
| Multi-User Isolation | 0.190 | 0.714 |
| Tone/Warmth | 0.762 | 0.952 |
| **Overall** | **0.476** | **0.850** |
