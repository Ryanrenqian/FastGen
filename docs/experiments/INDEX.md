# Experiment Hypothesis Index

Track all hypotheses in one place.

| Hypothesis ID | Title | Status | Grade | Branch | Worktree | Commit Range | Result Summary | Record |
|---------------|-------|--------|-------|--------|----------|--------------|----------------|--------|
| HYP-20260812-11 | Increasing TFD class-conditional sample count | supported | strong exploratory | feature/tfd-imagenet64-paper | required | 18ac5cb..bdc49f4 | 8/8/8 reaches official FID 1.3929 at 180.5k, versus paper 4/4/4 FID 1.58 | [link](./HYP-20260812-11.md) |
| HYP-20260810-09 | TFD ImageNet-64 official DMD2 LMDB retraining | preparing_data | exploratory | feature/tfd-imagenet64-paper | required | pending | official LMDB switch in progress | [link](./HYP-20260810-09.md) |
| HYP-20260807-05 | TFD ImageNet-64 reproduction | preparing_data | exploratory | feature/teacher-feature-drifting | required | pending | pending | [link](./HYP-20260807-05.md) |
| HYP-20260807-04 | TFD CIFAR-10 1000-step exploratory continuation | running | exploratory | feature/teacher-feature-drifting | required | 7c1136a75e6e | pending | [link](./HYP-20260807-04.md) |
| HYP-20260807-03 | TFD CIFAR-10 remote smoke training | completed | exploratory | feature/teacher-feature-drifting | required | 7c1136a75e6e | 100 steps stable; checkpoints saved | [link](./HYP-20260807-03.md) |
| HYP-20260807-01 | TBD | planned | exploratory | TBD | optional | TBD | TBD | [link](./HYP-20260807-01.md) |
| HYP-20260807-02 | TBD | planned | exploratory | TBD | required | TBD | TBD | [link](./HYP-20260807-02.md) |

## Usage Rules

- One row per hypothesis.
- Keep `Status`, `Commit Range`, and `Result Summary` current.
- `Record` must point to the detailed hypothesis markdown file.
- If a hypothesis is split, create new IDs; do not overwrite history.
| HYP-20260807-08 | TFD ImageNet-64 official-recipe reproduction | implementation validation | exploratory | `docs/experiments/HYP-20260807-08.md` |
