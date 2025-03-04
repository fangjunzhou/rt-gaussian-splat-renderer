import logging
from pathlib import Path
import pytest
import taichi as ti

from rtgs.scene import Scene


logger = logging.getLogger(__name__)

# Init taichi runtime.
ti.init(arch=ti.gpu, random_seed=42)
logger.info(f"Current Taichi backend: {ti.cfg.arch}")  # pyright: ignore


def test_scene_load():
    """Test load Gaussian splat scene."""
    scene = Scene()
    try:
        path = Path("tests/data/test.ply")
        scene.load_file(path)
    except Exception as e:
        pytest.fail("Load scene failed.")

# TODO: Handle illegal GS file.
