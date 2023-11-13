import unittest
import numpy as np
from gage.nfg.potential import potential
from gage.functions.monotonic import decreasing
from gage.utils.transforms import make_batched


class TestPotential(unittest.TestCase):
    batch_size = 2
    num_players = 3
    num_actions = 5
    min_r = 0
    max_r = 5
    num_points = 10
    seed = 1

    def check_potential(self, payoff_matrices, potential_funcs, weights):
        num_players = payoff_matrices.shape[1]
        strat_shape = payoff_matrices.shape[2:]
        batch_size = payoff_matrices.shape[0]

        for player in range(num_players):  # Loop over all players.
            player_actions = strat_shape[player]
            opp_strats = np.delete(strat_shape, player)
            for opp_strat in np.ndindex(*opp_strats):  # Loop over all opponent strats.
                comp_strat = list(opp_strat)
                comp_strat.insert(player, 0)
                comp_strat = tuple(comp_strat)
                comp_potential = potential_funcs(make_batched(comp_strat, batch_size))
                comp_u = payoff_matrices[(slice(None), player) + comp_strat]

                for action in range(1, player_actions):  # Verify that the utilities correspond to the potential.
                    joint_strat = list(opp_strat)
                    joint_strat.insert(player, action)
                    joint_strat = tuple(joint_strat)
                    new_potential = potential_funcs(make_batched(joint_strat, batch_size))
                    new_u = (new_potential - comp_potential) / weights[:, player] + comp_u
                    np.testing.assert_allclose(payoff_matrices[(slice(None), player) + joint_strat], new_u)

    def custom_potential_wikipedia(self, b1=2, b2=-1, w=3):
        """Create a potential function for the Wikipedia example.

        Note:
            https://en.wikipedia.org/wiki/Potential_game

        Args:
            b1 (int, optional): The coefficient for player 1. Defaults to 2.
            b2 (int, optional): The coefficient for player 2. Defaults to -1.
            w (int, optional): The coefficient for the interaction term. Defaults to 3.

        Returns:
            callable: A potential function.
        """

        def potential_func(strategy):
            step1 = strategy * np.array([b1, b2])
            step2 = np.sum(step1, axis=-1)
            step3 = w * np.prod(strategy, axis=-1)
            final = step2 + step3
            return final

        return potential_func

    def test_wikipedia_example(self):
        potential_func = self.custom_potential_wikipedia()
        payoff_matrices = potential(2,
                                    2,
                                    potential_func,
                                    batch_size=self.batch_size,
                                    min_r=self.min_r,
                                    max_r=self.max_r,
                                    seed=self.seed)

        self.check_potential(payoff_matrices, potential_func, np.ones((self.batch_size, 2)))

    def test_potential(self):
        rng = np.random.default_rng(self.seed)
        potential_func = decreasing(self.num_players, self.batch_size, num_points=self.num_actions, seed=self.seed)
        weights = rng.uniform(low=0, high=1, size=(self.batch_size, self.num_players))
        payoff_matrices = potential(self.num_players,
                                    self.num_actions,
                                    potential_func,
                                    batch_size=self.batch_size,
                                    weights=weights,
                                    min_r=self.min_r,
                                    max_r=self.max_r,
                                    seed=self.seed)

        self.check_potential(payoff_matrices, potential_func, weights)


if __name__ == '__main__':
    unittest.main()
