import unittest
from types import SimpleNamespace

from scripts.ttime_suite.export_moabb_for_deeptransfer import _apply_cli_overrides, _resolve_preset


class ExportOverrideTests(unittest.TestCase):
    def test_resolve_preset_supports_bciiv2b_alias(self) -> None:
        preset = _resolve_preset("bciiv2b")

        self.assertEqual(preset.dataset, "BNCI2014_004")
        self.assertEqual(preset.preprocess, "moabb")
        self.assertEqual(preset.fmin, 8.0)
        self.assertEqual(preset.fmax, 30.0)
        self.assertEqual(preset.tmin, 0.0)
        self.assertEqual(preset.tmax, 4.0)
        self.assertEqual(preset.resample, 250.0)
        self.assertIsNone(preset.sessions)

    def test_apply_cli_overrides_replaces_preset_fields(self) -> None:
        preset = _resolve_preset("physionetmi")
        args = SimpleNamespace(
            preprocess="moabb",
            fmin=4.0,
            fmax=40.0,
            tmin=0.5,
            tmax=3.5,
            resample=250.0,
            car=True,
            fir_order=80,
            fir_window="hann",
        )

        effective = _apply_cli_overrides(preset, args)

        self.assertEqual(effective.preprocess, "moabb")
        self.assertEqual(effective.fmin, 4.0)
        self.assertEqual(effective.fmax, 40.0)
        self.assertEqual(effective.tmin, 0.5)
        self.assertEqual(effective.tmax, 3.5)
        self.assertEqual(effective.resample, 250.0)
        self.assertTrue(effective.car)
        self.assertEqual(effective.paper_fir_order, 80)
        self.assertEqual(effective.paper_fir_window, "hann")

    def test_apply_cli_overrides_keeps_preset_when_override_missing(self) -> None:
        preset = _resolve_preset("physionetmi")
        args = SimpleNamespace(
            preprocess=None,
            fmin=None,
            fmax=None,
            tmin=None,
            tmax=None,
            resample=None,
            car=None,
            fir_order=None,
            fir_window=None,
        )

        effective = _apply_cli_overrides(preset, args)

        self.assertEqual(effective, preset)


if __name__ == "__main__":
    unittest.main()
