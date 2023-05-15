import unittest
import numpy as np
from gage.nfg.majority_voting import majority_voting


class TestMajorityVoting(unittest.TestCase):
    batch_size = 50
    min_r = 0
    max_r = 20
    seed = 0

    def check_majority_voting(self, num_players, num_candidates, payoff_matrices):
        for payoffs in payoff_matrices:
            player_utilities = []

            for winner in range(num_candidates):
                winner_votes = tuple(np.full(num_players, winner))
                player_utilities.append(payoffs[(slice(None),) + winner_votes])

            for idx in np.ndindex(payoffs[0].shape):
                winner = np.argmax(np.bincount(idx))
                np.testing.assert_array_equal(payoffs[(slice(None),) + idx], player_utilities[winner])

    def test_small(self):
        num_players = 2
        num_candidates = 3
        payoff_matrices = majority_voting(num_players,
                                          num_candidates,
                                          batch_size=self.batch_size,
                                          min_r=self.min_r,
                                          max_r=self.max_r,
                                          seed=self.seed)
        self.check_majority_voting(num_players, num_candidates, payoff_matrices)

    def test_large(self):
        num_players = 4
        num_candidates = 5
        payoff_matrices = majority_voting(num_players,
                                          num_candidates,
                                          batch_size=self.batch_size,
                                          min_r=self.min_r,
                                          max_r=self.max_r,
                                          seed=self.seed)
        self.check_majority_voting(num_players, num_candidates, payoff_matrices)


if __name__ == '__main__':
    unittest.main()
