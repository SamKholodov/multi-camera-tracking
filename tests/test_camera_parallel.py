import unittest
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.camera_manager import CameraManager


class FakeCap:
    def __init__(self, value):
        self.value = value

    def read(self):
        if self.value is None:
            return False, None
        return True, self.value


class TestCameraManagerParallelRead(unittest.TestCase):
    def test_parallel_read_preserves_camera_order(self):
        manager = object.__new__(CameraManager)
        manager.caps = [FakeCap("c006"), FakeCap(None), FakeCap("c008")]

        frames = manager.read_frames(parallel=True)

        self.assertEqual(frames, ["c006", None, "c008"])


if __name__ == "__main__":
    unittest.main()
