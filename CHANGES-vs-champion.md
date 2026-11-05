# Changes vs the champion (44cc7f5)

| ID | File / function | Behaviour change | Status |
|---|---|---|---|
| D0 | `scripts/image_trainer.py` (`TASK_EXTRAS`) | Removed the overrides keyed on task-ID prefixes (past R1 tasks). Recipes now depend only on family and data, never on task identity. | done (compliance, not an improvement; never submitted on its own) |
| D1 | `trainer/godtrainer/selection.py` (new); `engine.py` (`holdout_score`, `evaluate_candidates`, `_select`, delta-average, member/soup acceptance via `Trainer._better`); `train.py` (`--select-rule`, `--select-k`); `scripts/image_trainer.py` (`COMMON`) | Holdout uses 4 seeded noises per band (the evaluator's count; the champion used 1). Every candidate keeps its paired per-image vector. With ≥3 holdout images a candidate replaces the best only if its paired gain exceeds k·SE across images (k=1, a heuristic); otherwise a smoother/later state (deltaavg > ema > raw > base) wins only if its mean is not worse. With <3 images the champion's lower-mean rule applies. `--select-rule min` reproduces the champion. | implemented; CPU unit tests pass (12); **no GPU A/B yet**. Risk: ~4× holdout cases per evaluation (~2× on Flux, which used 2) until D3 budgets it. |
| D2 | `engine.py` (`Trainer.run`) | 2-fold cross-fit members, soup shipped only if the cross-fitted paired check favours it | planned |
| D3 | `engine.py` (`train_member`), `scripts/image_trainer.py` | Epoch-capped cosine horizon, time-grid evaluations, extra seed members, measured save margin | planned |
| D4 | `scripts/image_trainer.py` (`RECIPES`) | Per-family defaults re-fit on public tasks with D1–D3 in place | planned |
