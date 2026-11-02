import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import single
from single import (search_events, coverage, reference, sector_noise, binned, detrended, pair_aliases,
                    asymmetry, density_period, consistent_pair, common_mode_sigma)

SUN = {'radius': 1., 'logg': 4.44, 'teff': 5778., 'tmag': 10.}


def light_curve(events=(), depth=.003, hours=8., gaps=((), ), seed=1, noise=.0008):
    """Four 25-day sectors with 2-minute cadence and a mid-sector gap; box transits at `events`."""
    t = np.concatenate([2459000 + 30 * k + np.r_[np.arange(0, 12, 2 / 1440), np.arange(13, 25, 2 / 1440)]
                        for k in range(4)])
    sector = np.repeat(np.arange(1, 5), len(t) // 4)
    f = 1 + np.random.default_rng(seed).normal(0, noise, len(t))
    for e in events:
        f[np.abs(t - e) < hours / 48] -= depth
    return {'time_bjd': t, 'flux': f, 'sector': sector}


def ref_of(star, hours=8.):
    t, f, s = detrended(star['time_bjd'], star['flux'], star['sector'], hours / 24)
    return reference(t, f, s, sector_noise(t, f, s, hours / 24))


class SingleTests(unittest.TestCase):
    def test_single_event_found_with_epoch(self):
        events = search_events(light_curve([2459035.]))
        self.assertTrue(events)
        self.assertLess(abs(events[0]['epoch_bjd'] - 2459035.), 1 / 24)
        self.assertGreater(events[0]['duration_hours'], 5)

    def test_noise_only_gives_no_event(self):
        self.assertEqual(search_events(light_curve([])), [])

    def test_coverage_excludes_periods_with_a_flat_predicted_transit(self):
        star = light_curve([2459035., 2459095.])
        ref = ref_of(star)
        # P = 60 d: no other predicted transit inside the data. P = 30 d: 2459065 is covered and flat.
        excluded, _ = coverage(ref, 2459035., [60., 30.], 8 / 24, .003, [2459035., 2459095.])
        self.assertEqual(list(excluded), [False, True])

    def test_pair_aliases_keep_only_consistent_periods(self):
        star = light_curve([2459035., 2459095.])
        a = {'epoch_bjd': 2459035., 'duration_hours': 8., 'depth_ppm': 3000., 'snr': 20.}
        b = dict(a, epoch_bjd=2459095.)
        aliases, _ = pair_aliases(ref_of(star), a, b, SUN)
        periods = [x['period_days'] for x in aliases]
        self.assertIn(60., periods)
        self.assertNotIn(30., periods)
        self.assertTrue(consistent_pair(a, b))
        self.assertFalse(consistent_pair(a, dict(b, depth_ppm=9000.)))

    def test_asymmetry_flags_a_step(self):
        star = light_curve([])
        star['flux'][np.abs(star['time_bjd'] - 2459035.2) < .2] -= .004
        ref = ref_of(star)
        sigma, fraction = asymmetry(ref, 2459035., 16 / 24)
        self.assertGreater(sigma, single.ASYMMETRY_SIGMA)
        self.assertGreater(fraction, single.ASYMMETRY_FRACTION)
        self.assertLess(asymmetry(ref_of(light_curve([2459035.])), 2459035., 8 / 24)[0], 3)

    def test_density_period_for_the_sun(self):
        # A central transit of an Earth-size planet around the Sun lasts ~13 h at one year.
        self.assertAlmostEqual(density_period(13.1 / 24, 8.4e-5, SUN) / 365.25, 1, delta=.1)

    def test_common_mode_flags_a_shared_dip(self):
        bins = np.arange(int(2459030 / single.COMMON_BIN_DAYS), int(2459040 / single.COMMON_BIN_DAYS))
        centers = (bins + .5) * single.COMMON_BIN_DAYS
        values = np.random.default_rng(2).normal(0, 1, len(bins))
        shared = values - 5 * (np.abs(centers - 2459035.) < .15)
        star = {'time_bjd': np.arange(2459030, 2459040, .01), 'sector': np.ones(1000, int), 'meta': {}}
        original = single.ccd_of
        single.ccd_of = lambda star, sector: (1, 1)
        try:
            for values_, expect_high in ((shared, True), (values, False)):
                single._common = {'s1c1d1_bins': bins, 's1c1d1_median': values_,
                                  's1c1d1_stars': np.full(len(bins), 10)}
                sigma = common_mode_sigma(star, 1e4, 2459035., 8 / 24)
                self.assertEqual(sigma > single.COMMON_SIGMA, expect_high)
        finally:
            single.ccd_of, single._common = original, None


class CentroidTest(unittest.TestCase):
    def test_empirical_centroid_flags_only_a_real_shift(self):
        star = light_curve([2459035.], depth=.004)
        rng = np.random.default_rng(3)
        star['centr1'] = 500 + rng.normal(0, .002, len(star['time_bjd']))
        star['centr2'] = 800 + rng.normal(0, .002, len(star['time_bjd']))
        star['meta'] = {s: {'crowdsap': 1., 'star': SUN, 'target_pixel': [500., 800.]} for s in range(1, 5)}
        quiet, _ = single.centroid_single(star, [2459035.], 8 / 24, .004)
        star['centr1'][np.abs(star['time_bjd'] - 2459035.) < 4 / 24] += .01
        shifted, arcsec = single.centroid_single(star, [2459035.], 8 / 24, .004)
        self.assertLess(quiet, 3)
        self.assertGreater(shifted, single.CENTROID_SIGMA)
        self.assertGreater(arcsec, single.PIXEL_ARCSEC)


class DuoPipelineTest(unittest.TestCase):
    def test_two_transits_become_a_duo_with_the_true_period(self):
        star = light_curve([2459035., 2459095.], depth=.004)
        star['tic'] = 1
        star['meta'] = {s: {'crowdsap': .99, 'star': SUN} for s in range(1, 5)}
        original = single.COMMON_MODE
        single.COMMON_MODE = Path('/nonexistent/common_mode.npz')
        try:
            events = single.single_star_events(star, [], [])
        finally:
            single.COMMON_MODE = original
        duos = [e for e in events if e['search'] == 'duo']
        self.assertEqual(len(duos), 1)
        self.assertIn(60., [round(a['period_days'], 3) for a in duos[0]['aliases']])
        self.assertNotIn(30., [round(a['period_days'], 3) for a in duos[0]['aliases']])
        self.assertEqual(duos[0]['metrics']['verdict'], 'pass')


if __name__ == '__main__':
    unittest.main()
