import builtins
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.mot.association.association import linear_assignment


class TestLinearAssignment(unittest.TestCase):
    def _scipy_only(self, cost_matrix: np.ndarray) -> np.ndarray:
        real_import = builtins.__import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "lap":
                raise ImportError("lap unavailable")
            return real_import(name, globals, locals, fromlist, level)

        saved_lap = sys.modules.pop("lap", None)
        try:
            with patch("builtins.__import__", mock_import):
                return linear_assignment(cost_matrix)
        finally:
            if saved_lap is not None:
                sys.modules["lap"] = saved_lap

    def test_scipy_fallback_shape_and_dtype(self):
        cost = np.array([[1.0, 5.0, 9.0], [2.0, 0.5, 8.0], [4.0, 3.0, 6.0]])
        out = self._scipy_only(cost)
        self.assertEqual(out.shape, (3, 2))
        self.assertTrue(np.issubdtype(out.dtype, np.integer))

    def test_scipy_fallback_empty(self):
        out = self._scipy_only(np.zeros((0, 2)))
        self.assertEqual(out.shape, (0, 2))

    def test_scipy_fallback_usable_as_row_col_indices(self):
        cost = np.array([[10.0, 1.0], [1.0, 10.0]])
        out = self._scipy_only(cost)
        rows, cols = out[:, 0], out[:, 1]
        self.assertEqual(set(map(tuple, out)), {(0, 1), (1, 0)})
        self.assertEqual(cost[rows[0], cols[0]] + cost[rows[1], cols[1]], 2.0)


if __name__ == "__main__":
    unittest.main()
