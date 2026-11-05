# Changes vs the champion (44cc7f5)

| ID | File / function | Behaviour change | Status |
|---|---|---|---|
| D0 | `scripts/image_trainer.py` (`TASK_EXTRAS`) | Removed the overrides keyed on task-ID prefixes (past R1 tasks). Recipes now depend only on family and data, never on task identity. | done (compliance, not an improvement) |
| D1 | `trainer/godtrainer/engine.py` (`holdout_score`, `_select`) | 4-noise common-random-number holdout; paired one-SE checkpoint selection with a smoothness tie-break | planned |
| D2 | `engine.py` (`Trainer.run`) | 2-fold cross-fit members, soup shipped only if the cross-fitted paired check favours it | planned |
| D3 | `engine.py` (`train_member`), `scripts/image_trainer.py` | Epoch-capped cosine horizon, time-grid evaluations, extra seed members, measured save margin | planned |
| D4 | `scripts/image_trainer.py` (`RECIPES`) | Per-family defaults re-fit on public tasks with D1–D3 in place | planned |
