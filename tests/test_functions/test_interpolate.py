import unittest
import numpy as np
from itertools import product
from gage.functions.concave import concave_table
from gage.functions.interpolate import interpolate_table


class TestInterpolateTable(unittest.TestCase):
    dim = 2
    batch_size = 5
    min_y = 0
    max_y = 10
    num_points = 20
    seed = 0

    def test_interpolate_table(self):
        conc_tables = concave_table(dim=self.dim,
                                    batch_size=self.batch_size,
                                    min_y=self.min_y,
                                    max_y=self.max_y,
                                    num_points=self.num_points,
                                    seed=self.seed)
        conc_f = interpolate_table(conc_tables, batched=True)

        grid = list(product(*[range(self.num_points) for _ in range(self.dim)]))
        f_vals = conc_f(grid)
        t_vals = np.swapaxes([conc_tables[(slice(None),) + tuple(coords)] for coords in grid], 0, 1)
        np.testing.assert_equal(f_vals, t_vals)


if __name__ == '__main__':
    unittest.main()
