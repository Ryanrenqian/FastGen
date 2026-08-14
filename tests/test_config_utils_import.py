from pathlib import Path

from fastgen.configs.config_utils import import_config_from_python_file


def test_import_config_accepts_repo_relative_path():
    config = import_config_from_python_file(
        "fastgen/configs/experiments/WanI2V/config_tfd_wan22_5b.py"
    )
    assert config.model.feature_indices == [9, 19, 29]


def test_import_config_accepts_absolute_path():
    config_path = (
        Path(__file__).parents[1]
        / "fastgen/configs/experiments/WanI2V/config_tfd_wan22_5b.py"
    )
    config = import_config_from_python_file(str(config_path))
    assert config.model.feature_indices == [9, 19, 29]
