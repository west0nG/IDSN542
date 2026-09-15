import unittest
import numpy as np
import pandas as pd

from perceptron import Perceptron
from occupancy import evaluate, load_data, scale_features


class OccupancyTests(unittest.TestCase):
    def test_wrong_prediction_updates_weights_and_bias(self):
        # The seeded initial score for [2, -1] is positive; target 0 must lower it.
        X = np.array([[2.0, -1.0]])
        initial = np.random.RandomState(1).normal(0.0, 0.01, 2)
        ppn = Perceptron(eta=0.1, n_iter=1).fit(X, np.array([0]))
        np.testing.assert_allclose(ppn.w_, initial - 0.1 * X[0])
        self.assertAlmostEqual(ppn.b_, -0.1)
        self.assertEqual(ppn.errors_, [1])
        self.assertEqual(ppn.predict(X)[0], 0)

    def test_simple_separable_problem(self):
        X = np.array([[-2., 0.], [-1., 1.], [1., -1.], [2., 0.]])
        y = np.array([0, 0, 1, 1])
        before = X.copy()
        ppn = Perceptron().fit(X, y)
        np.testing.assert_array_equal(ppn.predict(X), y)
        np.testing.assert_array_equal(X, before)
        again = Perceptron().fit(X, y)
        np.testing.assert_array_equal(ppn.w_, again.w_)

    def test_scaling_uses_training_only(self):
        train = pd.DataFrame({"x": [0., 2.], "constant": [7., 7.]})
        test = pd.DataFrame({"x": [100.], "constant": [7.]})
        a, b, mean, scale = scale_features(train, test, ["x", "constant"])
        np.testing.assert_allclose(mean, [1., 7.])
        np.testing.assert_allclose(scale, [1., 1.])
        np.testing.assert_allclose(a[:, 0], [-1., 1.])
        np.testing.assert_allclose(b, [[99., 0.]])

    def test_false_empty_and_false_occupied_are_not_swapped(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([1, 0, 0, 0, 0, 1])
        score = evaluate(y, pred)
        self.assertEqual(score["false_empty"], 2)
        self.assertEqual(score["false_occupied"], 1)
        self.assertAlmostEqual(score["occupied_recall"], 1 / 3)
        self.assertAlmostEqual(score["balanced_accuracy"], 0.5)
        self.assertEqual(evaluate(y, np.zeros_like(y))["occupied_precision"], 0)

    def test_chronological_split(self):
        train, test = load_data()
        self.assertEqual(len(train) + len(test), 10129)
        self.assertLess(train.timestamp.max(), test.timestamp.min())
        self.assertTrue(set(train.timestamp.dt.date).isdisjoint(test.timestamp.dt.date))
        for frame in (train, test):
            self.assertEqual(set(frame.occupied), {0, 1})
            np.testing.assert_array_equal(frame.occupied, frame.Room_Occupancy_Count > 0)


if __name__ == "__main__":
    unittest.main()
