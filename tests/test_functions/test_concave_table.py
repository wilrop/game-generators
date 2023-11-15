import unittest
import numpy as np
from gage.functions.concave import concave_table


class TestConcaveTable(unittest.TestCase):
    dim = 2
    batch_size = 5
    min_y = 0
    max_y = 10
    num_points = 20
    seed = 0

    def test_concave_table(self):
        conc_tables = concave_table(
            dim=self.dim,
            batch_size=self.batch_size,
            min_y=self.min_y,
            max_y=self.max_y,
            num_points=self.num_points,
            seed=self.seed,
        )

        for conc_f in conc_tables:
            for d in range(self.dim):  # Undo the cumulative sum.
                up_slices = [slice(None)] * self.dim
                down_slices = [slice(None)] * self.dim
                up_slices[d] = slice(1, None)
                down_slices[d] = slice(None, -1)

                conc_f[tuple(up_slices)] -= conc_f[tuple(down_slices)]

            self.assertTrue(np.all(conc_f >= 0 - 1e-10))  # Check that the gradient never becomes negative.
            conc_f[(0,) * self.dim] = self.max_y - self.min_y

            for d in range(self.dim):  # Check that in each dimension, the derivative is decreasing.
                up_slices = [slice(None)] * self.dim
                down_slices = [slice(None)] * self.dim
                up_slices[d] = slice(1, None)
                down_slices[d] = slice(None, -1)
                np.testing.assert_array_less(0, conc_f[tuple(down_slices)] - conc_f[tuple(up_slices)])


if __name__ == "__main__":
    unittest.main()
