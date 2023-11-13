import unittest
import numpy as np
from gage.functions.concave import concave
from gage.nfg.bertrand_oligopoly import bertrand_oligopoly
from gage.utils.transforms import make_batched


class TestBertrandOligopoly(unittest.TestCase):
    batch_size = 2
    num_players = 3
    num_actions = 5
    min_y = 0
    max_y = 20
    num_points = 10
    seed = 0

    def test_bertrand_oligopoly(self):
        cost_funs = lambda x: np.full((self.batch_size, 1), 3)
        demand_funs = concave(1,
                              batch_size=self.batch_size,
                              min_y=self.min_y,
                              max_y=self.max_y,
                              num_points=self.num_points,
                              seed=self.seed)

        payoff_matrices = bertrand_oligopoly(self.num_players,
                                             self.num_actions,
                                             cost_fun=cost_funs,
                                             demand_fun=demand_funs,
                                             batch_size=self.batch_size)

        for batch_id, payoff_matrix in enumerate(payoff_matrices):
            for idx in np.ndindex(*([self.num_actions] * self.num_players)):
                min_players = np.nonzero(idx == np.min(idx))[0]
                remaining_players = np.delete(np.arange(self.num_players), min_players)
                m = len(min_players)
                min_price = np.min(idx) + 1
                demand = demand_funs(make_batched([min_price], self.batch_size))[batch_id]
                cost = cost_funs([demand / m])[batch_id]
                payoffs = min_price * (demand / m) - cost
                np.testing.assert_allclose(payoff_matrix[(min_players,) + idx], payoffs[0])
                if len(remaining_players) > 0:
                    np.testing.assert_array_equal(payoff_matrix[(remaining_players,) + idx], 0)


if __name__ == '__main__':
    unittest.main()
