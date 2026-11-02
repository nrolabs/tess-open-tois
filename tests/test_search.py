import unittest
import numpy as np
import torch
from data import aws_url, field_stars, masks_for, FIELD_SECTORS
from lightcurve import wrap
from neural import AveragePool
from search import (bls, flatten, inject, period_grid, match_known, tier, consistent, edge_dominated,
                    SOLAR_DURATION)
from vetting import refine, analyze

SUN = {'radius': 1., 'logg': 4.44, 'teff': 5778., 'tmag': 10.}


def star_with_transit(period=3., epoch=2459001., depth=.005, days=54, seed=4):
    t = 2459000 + np.arange(0, days, 2 / 1440)
    flux = 1 + np.random.default_rng(seed).normal(0, .001, len(t))
    flux[np.abs(wrap(t - epoch, period)) < .05] -= depth
    sector = np.where(t - t[0] < days / 2, 1, 2)
    return {'time_bjd': t, 'flux': flux, 'sector': sector, 'tic': 1,
            'meta': {1: {'crowdsap': .99, 'star': SUN}, 2: {'crowdsap': .99, 'star': SUN}}}


class SearchTests(unittest.TestCase):
    def test_period_grid_keeps_phase_drift_below_duration(self):
        baseline = 700.
        grid = period_grid(1., 20., baseline)
        self.assertTrue(np.all(np.diff(grid) > 0))
        self.assertGreaterEqual(grid[0], 1.)
        self.assertLessEqual(grid[-1], 20.)
        # Phase drift between neighboring periods over the baseline: <= half of the minimum duration / 2.
        drift = np.diff(grid) / grid[:-1] * baseline
        allowed = np.maximum(.02, .5 * SOLAR_DURATION(grid[:-1])) / 2
        self.assertTrue(np.all(drift <= allowed * 1.01))

    def test_strong_signal_reported_once(self):
        star = star_with_transit()
        found = bls(star['time_bjd'], flatten(star['time_bjd'], star['flux'], star['sector']), p_max=13)
        periods = [f['period_days'] for f in found]
        self.assertAlmostEqual(periods[0], 3., delta=.01)
        repeats = [p for p in periods[1:] if any(abs(p / (3. * r) - 1) < .02 for r in (1, 2, .5, 3, 1 / 3))]
        self.assertEqual(repeats, [])

    def test_bls_epoch_is_absolute_bjd(self):
        star = star_with_transit()
        best = bls(star['time_bjd'], flatten(star['time_bjd'], star['flux'], star['sector']), p_max=13)[0]
        self.assertLess(abs(wrap(best['epoch_bjd'] - 2459001., 3.)), .03)
        self.assertGreater(best['epoch_bjd'], 2.4e6)

    def test_few_transit_signal_is_detected(self):
        # Two transits in one sector used to be rejected by the SDE gate despite SNR > 40.
        t = 2459000 + np.arange(0, 27, 2 / 1440)
        flux = 1 + np.random.default_rng(5).normal(0, .001, len(t))
        flux[np.abs(wrap(t - 2459004., 11.)) < .1] -= .004
        found = bls(t, flatten(t, flux, np.ones(len(t))), p_max=13, max_signals=1)
        self.assertTrue(found)
        self.assertAlmostEqual(found[0]['period_days'], 11., delta=.1)

    def test_short_period_long_duration_not_aliased(self):
        # A 4.7 h transit at P = 1.07 d used to be found at 2P for lack of long durations.
        rng = np.random.default_rng(6)
        t = 2459000 + np.arange(0, 27, 2 / 1440)
        flux = 1 + rng.normal(0, .001, len(t))
        flux[np.abs(wrap(t - 2459000.5, 1.07)) < 4.7 / 48] -= .003
        found = bls(t, flatten(t, flux, np.ones(len(t))), p_max=13, max_signals=1)
        self.assertAlmostEqual(found[0]['period_days'], 1.07, delta=.01)

    def test_single_event_is_vetoed(self):
        t = np.arange(0, 27, 10 / 1440)
        f = np.random.default_rng(7).normal(0, .001, len(t))
        f[np.abs(t - 5.) < .1] -= .01  # one deep dip; the "second transit" at 15 d has nothing
        self.assertFalse(consistent(t, f, .001, 10., 5., .2))
        f[np.abs(t - 15.) < .1] -= .01
        self.assertTrue(consistent(t, f, .001, 10., 5., .2))

    def test_edge_veto(self):
        t = np.r_[np.arange(0, 13, 10 / 1440), np.arange(14, 27, 10 / 1440)]
        self.assertTrue(edge_dominated(t, 13.95, 12.95, .2))  # events at 12.95 (gap) and 26.9 (end of data)
        self.assertFalse(edge_dominated(t, 5., 3., .2))

    def test_refine_recovers_drifted_ephemeris(self):
        star = star_with_transit(period=3., epoch=2459001.)
        # Catalog period off by 0.002 d with a matching uncertainty: ~0.036 d drift at the end of the data.
        period, epoch = refine(star, 3.002, 2459001., 2.4, .002, .005)
        # Refinement resolution: drift of 1/8 duration at the ends of the data.
        cycles = (star['time_bjd'] - epoch) / period
        drift = np.abs(cycles * (period - 3.) + wrap(epoch - 2459001., 3.)).max()
        self.assertLess(drift, .1 / 8)
        self.assertLess(abs(period - 3.), abs(3.002 - 3.))

    def test_imprecise_catalog_ephemeris_is_rejected(self):
        star = star_with_transit()
        with self.assertRaisesRegex(ValueError, 'ephemeris'):
            analyze(star, 3., 2459001., 2.4, errors=(.05, .5))

    def test_injected_duration_follows_star(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 10, 1000)
        _, hours = inject(t, np.ones(len(t)), 365.25, 1., .0001, SUN, rng)
        self.assertGreater(hours, 9)
        self.assertLess(hours, 14)

    def test_match_known_harmonic(self):
        toi = [{'toi': 1.01, 'tfopwg_disp': 'CP', 'pl_orbper': 3., 'pl_tranmid': 2459001., 'pl_trandurh': 2.4}]
        self.assertEqual(match_known({'period_days': 6., 'epoch_bjd': 2459004., 'duration_hours': 2.4}, toi)['period_ratio'], 2)
        self.assertIsNone(match_known({'period_days': 4.1, 'epoch_bjd': 2459001., 'duration_hours': 2.4}, toi))

    def test_tier_rules(self):
        ok, bad = {'verdict': 'pass'}, {'verdict': 'fail'}
        self.assertTrue(tier(True, bad, None).startswith('rejected'))
        self.assertTrue(tier(True, ok, {'fpp': .001, 'nfpp': .2}).startswith('rejected'))
        self.assertTrue(tier(True, ok, {'fpp': .005, 'nfpp': .0001}).startswith('strong'))
        self.assertTrue(tier(True, ok, {'fpp': .012, 'nfpp': .0001}).startswith('candidate'))
        self.assertEqual(tier(False, ok, {'fpp': .001, 'nfpp': .0001}), 'inconclusive')
        self.assertEqual(tier(True, ok, None), 'candidate: no FPP')


class DataTests(unittest.TestCase):
    def test_masks_exclude_by_toi_number(self):
        tois = [{'toi': 1.01, 'pl_orbper': 3, 'pl_tranmid': 1, 'pl_trandurh': 2},
                {'toi': 1.02, 'pl_orbper': 5, 'pl_tranmid': 2, 'pl_trandurh': 3}]
        self.assertEqual(masks_for(tois, dict(tois[0])), [(5., 2., 3.)])
        self.assertEqual(len(masks_for(tois)), 2)

    def test_field_sample_is_stable_and_respects_minimum(self):
        products = {t: {str(s): ('f', 'u') for s in range(1, 1 + (t % 40) + 1)} for t in range(1, 400)}
        small = field_stars(products, set(), 5)
        large = field_stars(products, set(), 10)
        self.assertTrue(set(small) <= set(large))
        self.assertTrue(all(len(s) >= FIELD_SECTORS for s in large.values()))
        self.assertTrue(set(field_stars(products, set(large), 5)).isdisjoint(large))

    def test_aws_url(self):
        self.assertEqual(aws_url(278660115, '1', 'x.fits'),
                         'https://stpubdata.s3.amazonaws.com/tess/public/tid/s0001/0000/0002/7866/0115/x.fits')


class NetworkPoolTests(unittest.TestCase):
    def test_average_pool_matches_adaptive(self):
        for length in (30, 50, 61):
            x = torch.randn(2, 3, length)
            torch.testing.assert_close(AveragePool(8)(x), torch.nn.AdaptiveAvgPool1d(8)(x))


if __name__ == '__main__':
    unittest.main()
