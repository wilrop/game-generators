import unittest
import numpy as np
from itertools import chain, combinations
from game_generators.nfg.congestion import congestion
from game_generators.functions.monotonic import decreasing


class TestCongestion(unittest.TestCase):
    batch_size = 2
    min_r = 0
    max_r = 5
    num_points = 10
    seed = 1

    def check_congestion(self, num_players, num_facilities):
        num_actions = 2**num_facilities - 1
        payoff_funcs = decreasing(
            1, batch_size=self.batch_size * num_facilities, num_points=num_players + 1, seed=self.seed
        )
        payoff_funcs = [payoff_funcs[i :: self.batch_size] for i in range(self.batch_size)]
        payoff_matrices = congestion(num_players, num_facilities, payoff_funcs, batch_size=self.batch_size)
        actions = list(chain.from_iterable(combinations(range(num_facilities), r) for r in range(num_facilities + 1)))
        actions = actions[1:]

        strat_shape = payoff_matrices.shape[2:]

        for payoffs, funcs in zip(payoff_matrices, payoff_funcs):
            for joint_action in np.ndindex(*strat_shape):
                joint_payoff = payoffs[(slice(None),) + joint_action]
                subset_counts = np.bincount(joint_action, minlength=num_actions)
                facility_counts = np.zeros(num_facilities, dtype=int)
                for subset, count in enumerate(subset_counts):
                    for facility in actions[subset]:
                        facility_counts[facility] += count

                for player_payoff, player_subset in zip(joint_payoff, joint_action):
                    total = 0
                    for facility in actions[player_subset]:
                        total += funcs[facility]([facility_counts[facility]])
                    np.testing.assert_equal(player_payoff, total)

    def test_small(self):
        num_players = 2
        num_facilities = 2
        self.check_congestion(num_players, num_facilities)

    def test_large(self):
        num_players = 3
        num_facilities = 3
        self.check_congestion(num_players, num_facilities)


if __name__ == "__main__":
    unittest.main()
