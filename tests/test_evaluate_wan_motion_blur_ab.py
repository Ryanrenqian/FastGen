import numpy as np

from scripts.experiments.evaluate_wan_motion_blur_ab import (
    metrics,
    symmetric_motion_mask,
)


def test_symmetric_motion_mask_is_arm_order_invariant():
    control = np.zeros((3, 4, 4), dtype=np.float32)
    treatment = np.zeros_like(control)
    control[1:, 0, 0] = 1.0
    treatment[1:, 3, 3] = 1.0

    mask_ab = symmetric_motion_mask(control, treatment, 0.8)
    mask_ba = symmetric_motion_mask(treatment, control, 0.8)

    assert np.array_equal(mask_ab, mask_ba)
    assert mask_ab[0, 0]
    assert mask_ab[3, 3]


def test_metrics_separate_motion_amplitude_from_spatial_sharpness():
    frames = np.zeros((3, 7, 7, 3), dtype=np.float32)
    frames[1, 2:5, 2:5] = 1.0
    frames[2, 2:5, 3:6] = 1.0
    mask = np.ones((7, 7), dtype=bool)

    result = metrics(frames, mask)

    assert result["temporal_l1"] > 0
    assert result["motion_temporal_l1"] > 0
    assert result["motion_laplacian_energy"] > 0
    assert result["motion_laplacian_p90"] > 0
    assert result["motion_hf_temporal_delta"] > 0
