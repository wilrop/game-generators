import unittest
import numpy as np
from gage.nfg.zero_sum import zero_sum


class TestZeroSum(unittest.TestCase):
    n_players = 3
    num_actions = 10
    batch_size = 100
    seed = 0

    def check_sum(self, payoff_matrices):
        summed = np.sum(payoff_matrices, axis=1)
        np.testing.assert_almost_equal(summed, np.zeros_like(summed))

    def test_2p(self):
        payoff_matrices = zero_sum(2, self.num_actions, batch_size=self.batch_size, seed=self.seed)
        self.check_sum(payoff_matrices)

    def test_np(self):
        payoff_matrices = zero_sum(self.n_players, self.num_actions, batch_size=self.batch_size, seed=self.seed)
        self.check_sum(payoff_matrices)


if __name__ == '__main__':
    unittest.main()
