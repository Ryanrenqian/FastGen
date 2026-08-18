# Wan-VAE Temporal PCA and Clustering Analysis (2026-08-18)

- Quality: exploratory
- Server: `aliyun-47-116-39-128-1017`
- Branch: `analysis/wan-vae-temporal-web`
- Worktree: `/mnt/home/renqian/oneNFE/FastGen-wan-vae-analysis`
- Model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (`vae` subfolder)
- Dataset: `/mnt/dataset/demo5_dataset/manifests/demo5_clean_10k_f49_seed10.csv`
- Dataset snapshot: 10,000 videos (10,001 CSV lines including header)
- Sample selection: 24 rows, shuffled with seed 10
- Frames: zero-based 30, 31, 32, 33, 34 at 256x256
- Encoding: deterministic posterior mode, framewise
- Feature: 4x4 spatial grid pooling of each 48-channel latent frame
- Analysis: standardized absolute and first-frame-relative delta features; SVD PCA;
  k-means++ with 20 restarts and silhouette-based K selection over 2..8
- Output: `/mnt/home/renqian/imgGen/runtime/fastgen/analysis/wan_vae_temporal_web_20260818`

## Command

```bash
CUDA_VISIBLE_DEVICES=0 LOCAL_FILES_ONLY=true \
HF_HOME=/mnt/home/renqian/imgGen/runtime/MODEL/huggingface/hub \
/mnt/home/renqian/oneNFE/FastGen-tfd/.conda/envs/fastgen/bin/python \
scripts/experiments/analyze_wan_vae_temporal.py \
  --output-dir /mnt/home/renqian/imgGen/runtime/fastgen/analysis/wan_vae_temporal_web_20260818 \
  --num-samples 24 --seed 10 --start-frame 30 --frames 5 \
  --width 256 --height 256 --encode-mode framewise \
  --feature-pool spatial-grid --grid-size 4
```

## Results

The development-machine run encoded all 24 selected videos. Each video
produced a latent tensor of shape `[48, 5, 16, 16]`. The static report contains
24 RGB-frame previews, 24 absolute PCA trajectories, and 24 delta PCA
trajectories.

| Feature space | PC1+PC2 | Selected K | Silhouette | Mean adjacent speed |
|---|---:|---:|---:|---|
| absolute | 0.2793 | 8 | 0.4373 | 0.2841, 0.2875, 0.3078, 0.3038 |
| first-frame delta | 0.1900 | 3 | 0.6638 | 0.8696, 0.8809, 0.9531, 0.9577 |

Observations:

- Absolute latent clusters are stable within every five-frame clip. This space
  is dominated by video identity, appearance, and scene layout rather than the
  short temporal phase.
- Delta features have a clearer three-state organization. Sample 1 transitions
  from cluster `1` to `2` at source frame 33. Samples 16 and 19 transition from
  cluster `1` to `0` at source frame 32.
- The highest delta path lengths are samples 16 (6.171), 1 (5.461), and 19
  (5.268). Visual inspection confirms that sample 16 contains a clear hand and
  object manipulation during the analyzed window.
- Mean delta movement increases toward frames 33-34, suggesting slightly
  stronger latent change in the latter half of this sampled window.

## Validation

- Local algorithm and web-generation tests: `7 passed`.
- Development-machine smoke run: 2/2 videos encoded; report artifacts present.
- Formal run: 24/24 videos encoded without skips.
- Copied report: 89 files, 3.9 MB.
- Image integrity: all 72 per-sample preview/PCA images decoded successfully.
- HTML static image references: no missing files.

Local report copy:

`outputs/wan_vae_temporal_web_20260818/report.html`
