import unittest
import numpy as np
from data import eligible, LABELS
from lightcurve import prepare, views, VIEWS
from vetting import measure, analyze
from search import bls, flatten, inject

SUN = {'radius': 1., 'logg': 4.44, 'teff': 5778., 'tmag': 10.}


def light_curve(depth_odd=.01, depth_even=.01, secondary=0., seed=7):
    t = 2459000 + np.arange(0, 27, 2 / 1440)
    phase = (t - 2459001. + 1.5) % 3 - 1.5
    flux = 1 + np.random.default_rng(seed).normal(0, .001, len(t))
    odd = np.round((t - 2459001.) / 3) % 2 == 1
    flux[(np.abs(phase) < .05) & odd] -= depth_odd
    flux[(np.abs(phase) < .05) & ~odd] -= depth_even
    flux[np.abs(np.abs(phase) - 1.5) < .05] -= secondary
    return t, flux


def local_view(x):
    start = VIEWS[0][1] * VIEWS[0][2]
    return x[start:start + VIEWS[1][2]]


class ScienceTests(unittest.TestCase):
    def test_epoch_period_equivalence_and_flux_scale(self):
        t, flux = light_curve()
        a = views(prepare(t, flux, 3, 2459001., 2.4))
        b = views(prepare(t, flux * 1000, 3, 2459001. + 300, 2.4))
        np.testing.assert_allclose(a, b, atol=1e-8)
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(len(a), sum(c * n for _, c, n in VIEWS))
        self.assertLess(local_view(a)[30], -5)  # central transit preserved

    def test_missing_transit_is_not_a_negative_example(self):
        t = 2459000 + np.linspace(0, .2, 400)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            prepare(t, np.ones(400), 3, 2459001., 2.4)

    def test_unknown_candidates_excluded(self):
        self.assertNotIn('PC', LABELS)
        self.assertNotIn('APC', LABELS)
        self.assertFalse(eligible({'pl_orbper': float('nan'), 'pl_tranmid': 2459000, 'pl_trandurh': 2}))

    def test_clean_planet_passes_vetting(self):
        t, flux = light_curve(.003, .003)
        m = measure(prepare(t, flux, 3, 2459001., 2.4), SUN)
        self.assertEqual(m['verdict'], 'pass', m['flags'])
        self.assertGreater(m['snr'], 20)
        self.assertGreaterEqual(m['transits'], 8)
        self.assertAlmostEqual(m['depth_ppm'], 3000, delta=300)
        self.assertLess(m['odd_even_sigma'], 3)

    def test_odd_even_difference_fails(self):
        t, flux = light_curve(.01, .005)
        m = measure(prepare(t, flux, 3, 2459001., 2.4), SUN)
        self.assertEqual(m['flags'].get('odd_even'), 'fail')

    def test_secondary_eclipse_fails(self):
        t, flux = light_curve(.01, .01, secondary=.004)
        m = measure(prepare(t, flux, 3, 2459001., 2.4), SUN)
        self.assertEqual(m['flags'].get('secondary_eclipse'), 'fail')
        self.assertAlmostEqual(m['secondary_phase'], .5, delta=.02)

    def test_oversized_companion_fails(self):
        t, flux = light_curve(.3, .3)
        m = measure(prepare(t, flux, 3, 2459001., 2.4), SUN)
        self.assertEqual(m['flags'].get('too_large'), 'fail')

    def test_other_planet_is_masked(self):
        t, flux = light_curve()
        prep = prepare(t, flux, 3, 2459001., 2.4, masks=[(5., 2459002., 3.)])
        hidden = np.abs((prep['time'] - 2459002. + 2.5) % 5 - 2.5) < 3 / 24
        self.assertFalse(prep['valid'][hidden].any())

    def test_analyze_returns_finite_inputs(self):
        t, flux = light_curve()
        star = {'time_bjd': t, 'flux': flux, 'sector': np.ones(len(t), int),
                'centr1': np.random.default_rng(1).normal(500, .01, len(t)),
                'centr2': np.random.default_rng(2).normal(500, .01, len(t)),
                'meta': {1: {'crowdsap': .99, 'star': SUN, 'target_pixel': [500., 500.]}}, 'tic': 1}
        x, m = analyze(star, 3, 2459001., 2.4)
        self.assertTrue(np.isfinite(x).all())
        self.assertLess(m['centroid_sigma'], 3)

    def test_centroid_shift_detected(self):
        t, flux = light_curve()
        phase = (t - 2459001. + 1.5) % 3 - 1.5
        # 1% transit with a 0.02 px centroid shift: source ~2 px (42") from the target.
        c1 = np.random.default_rng(1).normal(500, .001, len(t))
        c1[np.abs(phase) < .05] -= .02
        star = {'time_bjd': t, 'flux': flux, 'sector': np.ones(len(t), int), 'centr1': c1,
                'centr2': np.random.default_rng(2).normal(500, .001, len(t)),
                'meta': {1: {'crowdsap': .99, 'star': SUN, 'target_pixel': [500., 500.]}}, 'tic': 1}
        _, m = analyze(star, 3, 2459001., 2.4)
        self.assertEqual(m['flags'].get('centroid_offset'), 'fail')
        self.assertAlmostEqual(m['source_offset_arcsec'], 42, delta=8)

    def test_bls_finds_injected_transit(self):
        rng = np.random.default_rng(3)
        t = 2459000 + np.arange(0, 27, 2 / 1440)
        flux, hours = inject(t, 1 + rng.normal(0, .001, len(t)), 4.2, 2459001.3, .002, SUN, rng)
        found = bls(t, flatten(t, flux, np.ones(len(t))), p_max=13, max_signals=1)
        self.assertTrue(found)
        self.assertAlmostEqual(found[0]['period_days'], 4.2, delta=.04)
        self.assertGreater(hours, 1)


if __name__ == '__main__':
    unittest.main()
