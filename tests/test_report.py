import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import report


class ReportTests(unittest.TestCase):
    def test_beta_is_one_for_white_noise_and_larger_for_red(self):
        rng = np.random.default_rng(0)
        t = np.arange(0, 5, 2 / 1440)
        white = rng.normal(0, 1e-3, len(t))
        red = white + 1e-3 * np.sin(2 * np.pi * t / .2)
        self.assertLess(report.beta_factor(t, white, .1), 1.2)
        self.assertGreater(report.beta_factor(t, red, .1), 1.5)

    def test_derived_duration_of_a_central_transit(self):
        # Sun-like star (1.41 g/cm³), P = 365.25 d, k ≈ 0.009: central transit lasts ~13 h.
        samples = {'k': np.full(10, .0092), 'b': np.zeros(10), 'period': np.full(10, 365.25),
                   'log_rho': np.full(10, np.log10(1.41)), 'q1': np.full(10, .36), 'q2': np.full(10, .3)}
        out = report.derived(samples, {'rad': 1., 'e_rad': 1e-6, 'Teff': 5772., 'e_Teff': 1e-6},
                             np.random.default_rng(0))
        self.assertAlmostEqual(np.median(out['duration_hours']), 13.1, delta=.3)
        self.assertAlmostEqual(np.median(out['a_au']), 1., delta=.01)
        self.assertAlmostEqual(np.median(out['radius_earth']), 1.004, delta=.01)

    def test_model_reproduces_injected_depth(self):
        t = np.linspace(-.2, .2, 2000)
        model = report.Model(t, np.ones_like(t))
        # theta = [t0, P, k, b, log_rho, q1, q2, f0, log_sigma]
        curve = model.light_curve([0., 3., .1, 0., np.log10(1.41), .36, .3, 0., -7.])
        self.assertAlmostEqual(1 - curve.min(), .01 * 1.15, delta=.002)
        self.assertAlmostEqual(curve[0], 1., places=6)

    def test_latex_escaping_and_formatting(self):
        self.assertEqual(report.tex('a_b 5% #1'), 'a\\_b 5\\% \\#1')
        self.assertEqual(report.fmt({'value': 2.5, 'minus': .12, 'plus': .15}), '$2.50^{+0.15}_{-0.12}$')
        self.assertEqual(report.fmt(None), '--')


if __name__ == '__main__':
    unittest.main()
