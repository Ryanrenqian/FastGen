# Teacher-Feature Drifting

`TFDModel` implements one-step diffusion distillation from
[Teacher-Feature Drifting](https://arxiv.org/abs/2605.07327). It uses a frozen diffusion teacher both as the
source checkpoint and as the feature encoder. The objective combines multi-radius drifting on moderately noised
teacher features with the paper's anchor-margin coverage loss.

The ImageNet-64 EDM recipe can be started with:

```bash
torchrun --nproc_per_node=8 train.py \
    --config=fastgen/configs/experiments/EDM/config_tfd_in64.py \
    - trainer.ddp=True log_config.name=tfd_in64
```

For a faster algorithm check, use `fastgen/configs/experiments/EDM/config_tfd_cifar10.py` with the same command.

The default recipe generates four student samples per class condition and maintains a four-image CPU reference bank
per class. A loader can instead provide `positive` with shape `[B, N_pos, C, H, W]`; this is recommended when exact
grouped positives or teacher-generated positives are available.
The EDM experiment recipes use FP16 autocast with FP32 parameters, a 500-step warmup, AdamW weight decay 0.01,
and gradient clipping at norm 10, matching the paper's ImageNet training protocol.

The main method-specific settings are `feature_indices`, `feature_noise_level`, `feature_pool_size`, `drift_radii`,
`generated_samples_per_condition`, `positive_samples_per_condition`, and the three `anchor_*` values. The current
implementation is intentionally limited to one-step students.
