import unittest
import numpy as np
from game_generators.functions.concave import concave
from game_generators.nfg.bertrand_oligopoly import bertrand_oligopoly


class TestBertrandOligopoly(unittest.TestCase):
    batch_size = 4
    num_players = 3
    num_actions = 5
    min_y = 0
    max_y = 20
    num_points = 10
    seed = 0

    def test_bertrand_oligopoly(self):
        cost_funs = [lambda x: 3 for _ in range(self.batch_size)]
        demand_funs = concave(
            1,
            batch_size=self.batch_size,
            min_y=self.min_y,
            max_y=self.max_y,
            num_points=self.num_points,
            seed=self.seed,
        )

        payoff_matrices = bertrand_oligopoly(
            self.num_players, self.num_actions, cost_funs=cost_funs, demand_funs=demand_funs, batch_size=self.batch_size
        )

        for payoff_matrix, cost_fun, demand_fun in zip(payoff_matrices, cost_funs, demand_funs):
            for idx in np.ndindex(*([self.num_actions] * self.num_players)):
                min_players = np.nonzero(idx == np.min(idx))[0]
                remaining_players = np.delete(np.arange(self.num_players), min_players)
                m = len(min_players)
                min_price = np.min(idx) + 1
                demand = demand_fun([min_price])
                cost = cost_fun(demand / m)
                payoffs = min_price * (demand / m) - cost
                np.testing.assert_allclose(payoff_matrix[(min_players,) + idx], payoffs[0])
                if len(remaining_players) > 0:
                    np.testing.assert_array_equal(payoff_matrix[(remaining_players,) + idx], 0)


if __name__ == "__main__":
    unittest.main()
