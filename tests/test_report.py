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

    def test_gpu_model_matches_batman_and_exact_exposure_average(self):
        import batman
        t = np.linspace(-.12, .12, 1201)
        exposure = report.EXPOSURE_DAYS
        for k, b, q1, q2, log_rho, period in ((.1, 0., .36, .3, .15, 3.), (.084, .832, .33, .46, .196, 2.31),
                                               (.01, .3, .5, .2, .5, 10.), (.15, .5, .9, .1, 1., 1.5)):
            model = report.Model(t, np.ones_like(t), None, (.6 * k, 1.6 * k))
            theta = [0., period, k, b, log_rho, q1, q2, 0., -7.]
            p = batman.TransitParams()
            a = report.a_over_r(10 ** log_rho, period)
            p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0., period, k, a, np.degrees(np.arccos(b / a)), 0., 90.
            p.u, p.limb_dark = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)], 'quadratic'
            # Instantaneous flux: exact geometry against the analytic Mandel & Agol model.
            reference = batman.TransitModel(p, t).light_curve(p)
            self.assertLess(np.max(np.abs(model.light_curve(theta, t) - reference)), 1e-7)
            # 120 s average: against a 201-point midpoint average (batman's 7-point linspace is coarser).
            fine = ((np.arange(201) + .5) / 201 - .5) * exposure
            exact = model.light_curve(theta, (t[:, None] + fine).ravel()).reshape(len(t), -1).mean(1)
            self.assertLess(np.max(np.abs(model.light_curve(theta) - exact)), 1e-6)

    def test_latex_escaping_and_formatting(self):
        self.assertEqual(report.tex('a_b 5% #1'), 'a\\_b 5\\% \\#1')
        self.assertEqual(report.fmt({'value': 2.5, 'minus': .12, 'plus': .15}), '$2.50^{+0.15}_{-0.12}$')
        self.assertEqual(report.fmt(None), '--')


if __name__ == '__main__':
    unittest.main()
