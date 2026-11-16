import contextlib
import io
import unittest

from LibMTL.config import LibMTL_args, prepare_args


class SchedulerConfigTest(unittest.TestCase):
    def _prepare(self, scheduler: str):
        params = LibMTL_args.parse_args(
            ["--weighting", "EW", "--arch", "HPS", "--scheduler", scheduler]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_args(params)[2]

    def test_step_scheduler_keeps_step_parameters(self):
        self.assertEqual(
            self._prepare("step"),
            {"scheduler": "step", "step_size": 100, "gamma": 0.5},
        )

    def test_cos_scheduler_returns_cosine_parameters(self):
        self.assertEqual(
            self._prepare("cos"),
            {"scheduler": "cos", "T_max": 100, "eta_min": 0.0},
        )

    def test_exp_scheduler_returns_exponential_parameters(self):
        self.assertEqual(
            self._prepare("exp"),
            {"scheduler": "exp", "gamma": 0.5},
        )


if __name__ == "__main__":
    unittest.main()
