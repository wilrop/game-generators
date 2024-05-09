import unittest
import numpy as np
from game_generators.functions.monotonic import increasing_table


class TestMonotonicTable(unittest.TestCase):
    dim = 2
    batch_size = 5
    min_y = 0
    max_y = 10
    num_points = 20
    seed = 0

    def check_increasing(self, table):
        dim = len(table.shape)
        padded_table = np.pad(table, (1, 0), mode="constant", constant_values=0)
        for idx in np.ndindex(table.shape):
            idx = tuple([i + 1 for i in idx])
            lower_bounds = []
            for d in range(dim):
                constraint_idx = tuple([idx[i] if i != d else idx[i] - 1 for i in range(dim)])
                lower_bounds.append(padded_table[constraint_idx])
            assert padded_table[idx] >= np.max(lower_bounds), f"Table is not increasing: {table}"

    def test_monotonic_table(self):
        for method in ["cumsum", "max_add"]:
            monotonic_tables = increasing_table(
                dim=self.dim,
                batch_size=self.batch_size,
                min_y=self.min_y,
                max_y=self.max_y,
                num_points=self.num_points,
                seed=self.seed,
                method=method,
            )

            for table in monotonic_tables:
                self.check_increasing(table)


if __name__ == "__main__":
    unittest.main()
