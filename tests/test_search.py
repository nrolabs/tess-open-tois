import unittest
import numpy as np
import torch
from data import aws_url, field_stars, masks_for, FIELD_SECTORS
from lightcurve import wrap
from neural import AveragePool
from search import (bls, flatten, inject, period_grid, match_known, known_masks, tier, consistent, edge_dominated,
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
        f = np.random.default_rng(8).normal(0, .001, len(t))
        dips = lambda period, epoch: np.where(np.abs(wrap(t - epoch, period)) < .1, -.005, 0.)
        # Both events at edges (12.95 next to the gap, 26.9 at the end of data): vetoed.
        self.assertTrue(edge_dominated(t, f + dips(13.95, 12.95), .001, 13.95, 12.95, .2))
        # No event near an edge: kept.
        self.assertFalse(edge_dominated(t, f + dips(5., 3.), .001, 5., 3., .2))
        # A real planet with one of its events at the gap: the other events carry it, kept.
        self.assertFalse(edge_dominated(t, f + dips(4.95, 3.05), .001, 4.95, 3.05, .2))

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
        toi = [{'source': 'TOI', 'name': 'TOI 1.01', 'toi': 1.01, 'disposition': 'CP', 'period': 3.,
                'epoch': 2459001., 'duration_hours': 2.4, 'transiting': True}]
        match = match_known({'period_days': 6., 'epoch_bjd': 2459004., 'duration_hours': 2.4}, toi)
        self.assertEqual((match['period_ratio'], match['toi']), (2, 1.01))
        self.assertIsNone(match_known({'period_days': 4.1, 'epoch_bjd': 2459001., 'duration_hours': 2.4}, toi))
        # Same period, other phase: a second planet in resonance is not the catalogued one.
        self.assertIsNone(match_known({'period_days': 3., 'epoch_bjd': 2459002.5, 'duration_hours': 2.4}, toi))

    def test_match_known_ctoi_and_planets(self):
        event = {'period_days': 20., 'epoch_bjd': 2459010., 'duration_hours': 4.}
        single = {'source': 'CTOI', 'name': 'CTOI 5.01', 'disposition': 'PC', 'period': None,
                  'epoch': 2459070.05, 'duration_hours': None, 'transiting': True}
        self.assertEqual(match_known(event, [single])['name'], 'CTOI 5.01')
        self.assertIsNone(match_known(event, [dict(single, epoch=2459075.)]))
        rv = {'source': 'planet', 'name': 'HD 1 b', 'disposition': 'published', 'period': 10.02,
              'epoch': 2450000., 'duration_hours': None, 'transiting': False}
        match = match_known(event, [rv])
        self.assertEqual((match['period_ratio'], match['phase_checked']), (2, False))
        self.assertEqual(known_masks([single, rv, dict(rv, name='b', transiting=True, duration_hours=3.)]),
                         [(10.02, 2450000., 3.)])
        # The same planet listed twice (e.g. TOI and TCE): neither entry may mask the event itself.
        twin = dict(rv, name='TCE', transiting=True, duration_hours=4., period=20., epoch=2459010.)
        other = dict(twin, name='c', period=7.3)
        self.assertEqual(known_masks([twin, dict(twin, name='TOI'), other], event), [(7.3, 2459010., 4.)])
        # A stale ephemeris (drift beyond a quarter period) still matches, by period alone.
        tce = {'source': 'TCE', 'name': 'TCE s0001-s0001 1-01', 'disposition': None, 'period': 20.,
               'epoch': 2458330., 'period_err': .05, 'epoch_err': .001, 'duration_hours': 4., 'transiting': True}
        self.assertEqual(match_known(dict(event, epoch_bjd=2459017.), [tce])['phase_checked'], False)
        self.assertIsNone(match_known(dict(event, epoch_bjd=2459017.), [dict(tce, period_err=1e-5)]))

    def test_json_safe_reports(self):
        from search import json_safe
        value = json_safe({'a': float('nan'), 'b': [np.inf, np.float64(2.)], 'c': np.int64(3), 'd': np.bool_(True)})
        self.assertEqual(value, {'a': None, 'b': [None, 2.], 'c': 3, 'd': True})

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
