import sys
import unittest
from unittest import mock

import numpy as np

from scripts.ttime_suite.kooptta_reference_plain_tta import streaming_normalize_trials_reference
from scripts.ttime_suite.run_suite_loso_kooptta_reference import parse_args


class KoopTTAReferenceRunnerTests(unittest.TestCase):
    def test_streaming_normalization_does_not_peek_future_trials(self) -> None:
        x = np.array([[[1.0]], [[3.0]]], dtype=np.float32)
        session_ids = np.array(["0", "0"])
        run_ids = np.array(["0", "0"])
        trial_ids = np.array([0, 1], dtype=np.int64)

        normalized = streaming_normalize_trials_reference(
            x,
            session_ids=session_ids,
            run_ids=run_ids,
            trial_ids=trial_ids,
        )

        self.assertAlmostEqual(float(normalized[0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(normalized[1, 0, 0]), 2000.0)

    def test_parse_args_uses_kooptta_like_note_defaults(self) -> None:
        argv = [
            "prog",
            "--data-dir",
            "/tmp/data",
            "--out-dir",
            "/tmp/out",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.note_memory_size, 50)
        self.assertEqual(args.note_replay_batch_size, 32)
        self.assertTrue(args.use_streaming_normalization)
        self.assertIn("pl_dteeg_ref", args.methods)
        self.assertIn("sar_dteeg_ref", args.methods)
        self.assertIn("delta_dteeg_ref", args.methods)
        self.assertIn("cotta_dteeg_ref", args.methods)
        self.assertIn("ttime_dteeg_ref", args.methods)


if __name__ == "__main__":
    unittest.main()
