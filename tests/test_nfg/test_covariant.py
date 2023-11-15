import unittest
import numpy as np
from game_generators.nfg.covariant import covariant


class TestCovariant(unittest.TestCase):
    n_players = 3
    num_actions = 10
    batch_size = 100
    mean = 0
    std = 1
    cov = 1
    seed = 0

    def check_covariance(self, n_players, payoff_matrices):
        cov_matrix = np.full((n_players, n_players), self.cov)
        np.fill_diagonal(cov_matrix, self.std)

        for payoffs in payoff_matrices:
            m = np.reshape(payoffs, (n_players, -1))
            test_cov = np.cov(m)
            np.testing.assert_allclose(cov_matrix, test_cov, rtol=0.4)

    def test_2p(self):
        payoff_matrices = covariant(
            2, self.num_actions, batch_size=self.batch_size, mean=self.mean, std=self.std, cov=self.cov, seed=self.seed
        )
        self.check_covariance(2, payoff_matrices)

    def test_np(self):
        payoff_matrices = covariant(
            self.n_players,
            self.num_actions,
            batch_size=self.batch_size,
            mean=self.mean,
            std=self.std,
            cov=self.cov,
            seed=self.seed,
        )
        self.check_covariance(self.n_players, payoff_matrices)


if __name__ == "__main__":
    unittest.main()
