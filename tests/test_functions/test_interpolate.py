import unittest
import numpy as np
from itertools import product
from game_generators.functions.concave import concave_table
from game_generators.functions.interpolate import interpolate_table


class TestInterpolateTable(unittest.TestCase):
    dim = 2
    batch_size = 5
    min_vec = np.array([1, 2])
    max_vec = np.array([5, 10])
    min_y = 0
    max_y = 10
    num_points = 20
    seed = 0

    def test_interpolate_table(self):
        conc_tables = concave_table(
            dim=self.dim,
            min_y=self.min_y,
            max_y=self.max_y,
            batch_size=self.batch_size,
            num_points=self.num_points,
            seed=self.seed,
        )
        conc_f = interpolate_table(conc_tables, batched=True)

        grid = list(product(*[range(self.num_points) for _ in range(self.dim)]))
        f_vals = [f(grid) for f in conc_f]
        t_vals = np.swapaxes([conc_tables[(slice(None),) + tuple(coords)] for coords in grid], 0, 1)
        np.testing.assert_equal(f_vals, t_vals)

    def test_interpolate_custom_input(self):
        conc_tables = concave_table(
            dim=self.dim,
            min_y=self.min_y,
            max_y=self.max_y,
            batch_size=self.batch_size,
            num_points=self.num_points,
            seed=self.seed,
        )
        conc_f = interpolate_table(conc_tables, min_vec=self.min_vec, max_vec=self.max_vec, batched=True)
        rng = np.random.default_rng(self.seed)
        samples = rng.uniform(self.min_vec, self.max_vec, size=(self.num_points, self.dim))

        f_vals = [f(samples) for f in conc_f]
        assert np.array(f_vals).shape == (self.batch_size, self.num_points)


if __name__ == "__main__":
    unittest.main()
