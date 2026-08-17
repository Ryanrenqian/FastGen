# DriftWorld-style TI2V training

`DriftWorldModel` trains a one-step conditional generator by drawing several
outputs for the same condition and moving them along a normalized drifting
field toward the matching data video. The Wan2.2 configuration uses text and
the VAE-encoded first frame as the shared TI2V condition.

The default objective combines two batch-independent fields after removing the
TI2V conditioning latent slot: a local field at every latent cell and the
Bridge DriftWorld DINOv3 field. The DINOv3 field
differentiably decodes Wan latents, extracts normalized ViT-B/16 patch tokens
from blocks 2, 5, and 8, and applies the reference motion-token weighting. The
candidates act as mutually repulsive samples and the data video is the positive
sample. Only the DINOv3 fields add the preceding real frame as a fixed
static-transition negative; the latent field has no extra fixed negative. This
preserves the original latent-spatial signal while adding
perceptual semantics. The optional trajectory-block field remains implemented
for ablations but is disabled in the formal configuration. It does not import Bridge's
action-specific U-Net or autoregressive self-forcing stage; Wan produces the
requested video in one forward pass.

The DINOv3 backbone is frozen but is not evaluated under `no_grad` for generated
videos, so gradients flow through DINOv3 and the frozen Wan VAE decoder into the
generator. Both paths use activation checkpointing. The default ModelScope
checkpoint is `facebook/dinov3-vitb16-pretrain-lvd1689m`; its safetensors keys
are strictly mapped to the pinned official DINOv3 backbone implementation.

EMA is disabled in the 5B FSDP recipe because FastGen keeps EMA replicas
unsharded. Enabling it would place another full 5B model on every rank.

The ready-to-run experiment is
`fastgen/configs/experiments/WanI2V/config_driftworld_wan22_5b.py`. Update its
CSV manifest path, then launch distributed training with:

```bash
torchrun --nproc_per_node=8 train.py \
  --config=fastgen/configs/experiments/WanI2V/config_driftworld_wan22_5b.py
```

The default run uses the clean Demo5 10k manifest at
`/mnt/dataset/demo5_dataset/manifests/demo5_clean_10k_f49_seed10.csv`, trains
on five adjacent RGB frames at indices 30/31/32/33/34 at 256x256, skipping
the initially static portion of each video, and initializes the student from
`Wan-AI/Wan2.2-TI2V-5B-Diffusers`. Each RGB frame is independently encoded
into one Wan latent frame and each generated latent frame is independently
decoded. Each condition draws 64 candidates, matching Bridge DriftWorld's
`n_neg=64`.
Only the first candidate per condition is returned to the generic W&B callback;
the experiment records generated and ground-truth videos every 500 steps while
keeping scalar metrics at the launcher's 10-step interval.
