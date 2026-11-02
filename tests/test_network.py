import unittest
import numpy as np
import torch
from neural import TransitCNN, INPUT_SIZE, inputs, outer_folds, choose_threshold, wilson


class NetworkTests(unittest.TestCase):
    def test_each_star_tested_once_without_leakage(self):
        groups = np.arange(200)
        y = groups % 2
        tested = []
        for train, validation, test in outer_folds(y, groups):
            tested += list(test)
            for a, b in ((train, validation), (train, test), (validation, test)):
                self.assertFalse(set(groups[a]) & set(groups[b]))
        self.assertEqual(sorted(tested), list(range(200)))

    def test_no_threshold_when_precision_is_unsupported(self):
        self.assertIsNone(choose_threshold(np.array([1, 0] * 10), np.full(20, .5)))
        self.assertIsNone(choose_threshold(np.ones(4), np.ones(4)))

    def test_threshold_is_selected_from_validation(self):
        y = np.array([0] * 10 + [1] * 10)
        p = np.arange(20) / 20
        self.assertEqual(choose_threshold(y, p, .95), .5)
        self.assertEqual(wilson(0, 0), None)
        self.assertLess(wilson(5, 5)[0], .95)

    def test_non_finite_input_rejected(self):
        X = np.zeros((1, INPUT_SIZE))
        X[0, 3] = np.nan
        with self.assertRaises(ValueError):
            inputs(X)

    def test_shape_and_gradient_are_finite(self):
        # Este teste de contrato usa CPU; treino real exige CUDA.
        model = TransitCNN()
        X = torch.tensor(inputs(np.random.default_rng(8).normal(size=(3, INPUT_SIZE))))
        output = model(X)
        self.assertEqual(tuple(output.shape), (3,))
        output.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
