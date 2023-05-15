import unittest
import numpy as np
from gage.nfg.hawk_dove import hawk_dove


class TestHawkDove(unittest.TestCase):
    batch_size = 1000
    min_r = 0
    max_r = 20
    seed = 0

    def test_hawk_dove(self):
        payoff_matrices = hawk_dove(batch_size=self.batch_size,
                                    min_r=self.min_r,
                                    max_r=self.max_r,
                                    seed=self.seed)

        a1 = payoff_matrices[:, 0, 1, 0]
        b1 = payoff_matrices[:, 0, 0, 0]
        c1 = payoff_matrices[:, 0, 0, 1]
        d1 = payoff_matrices[:, 0, 1, 1]

        a2 = payoff_matrices[:, 1, 0, 1]
        b2 = payoff_matrices[:, 1, 0, 0]
        c2 = payoff_matrices[:, 1, 1, 0]
        d2 = payoff_matrices[:, 1, 1, 1]

        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(d1, d2)

        np.testing.assert_array_less(b1, a1)
        np.testing.assert_array_less(c1, b1)
        np.testing.assert_array_less(d1, c1)


if __name__ == '__main__':
    unittest.main()
