import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixels import difference_image, fit_source, transit_centers


def cube(source, depth=.01, period=3., epoch=2459001., seed=5):
    """11×11 stamp: target at (5, 5), a neighbor a quarter as bright at (7, 5); `source` dims."""
    t = 2459000 + np.arange(0, 25, 2 / 1440)
    yy, xx = np.mgrid[0:11, 0:11]
    star = lambda x0, y0, f: f * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * .9 ** 2))
    target, neighbor = star(5, 5, 1e4), star(7, 5, 2.5e3)
    dims = (np.abs((t - epoch + period / 2) % period - period / 2) < .05)[:, None, None]
    images = np.where(dims, (target * (1 - depth) + neighbor) if source == 'target'
                      else (target + neighbor * (1 - 4 * depth)), target + neighbor)
    rng = np.random.default_rng(seed)
    return t, images + rng.normal(0, 5, images.shape) + 50


class PixelTests(unittest.TestCase):
    def locate(self, source):
        t, images = cube(source)
        image = difference_image(t, images, transit_centers(t, 3., 2459001.), 2.4 / 24)
        return fit_source(image['difference'], image['error'])

    def test_transit_on_target_is_located_on_target(self):
        fit = self.locate('target')
        self.assertLess(np.hypot(fit['x'] - 5, fit['y'] - 5), .2)
        self.assertGreater(fit['snr'], 10)

    def test_transit_on_neighbor_is_located_on_neighbor(self):
        fit = self.locate('neighbor')
        self.assertLess(np.hypot(fit['x'] - 7, fit['y'] - 5), .2)

    def test_single_event_centers(self):
        self.assertEqual(list(transit_centers(np.array([0., 50.]), 1e6, 20.)), [20.])


if __name__ == '__main__':
    unittest.main()
