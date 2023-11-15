import unittest
import numpy as np
from gage.utils.payoffs import create_congestion_potential_func
from gage.nfg.potential import potential
from gage.nfg.congestion import congestion
from gage.functions.monotonic import decreasing


class TestPotential(unittest.TestCase):
    batch_size = 2
    min_r = 0
    max_r = 5
    num_points = 10
    seed = 1

    def check_potential(self, payoff_matrices, potential_funcs, weights):
        num_players = payoff_matrices.shape[1]
        strat_shape = payoff_matrices.shape[2:]

        for player in range(num_players):  # Loop over all players.
            player_actions = strat_shape[player]
            opp_strats = np.delete(strat_shape, player)
            for opp_strat in np.ndindex(*opp_strats):  # Loop over all opponent strats.
                comp_strat = list(opp_strat)
                comp_strat.insert(player, 0)
                comp_strat = tuple(comp_strat)
                comp_potential = np.array([f(comp_strat) for f in potential_funcs])
                comp_u = payoff_matrices[(slice(None), player) + comp_strat]

                for action in range(1, player_actions):  # Verify that the utilities correspond to the potential.
                    joint_strat = list(opp_strat)
                    joint_strat.insert(player, action)
                    joint_strat = tuple(joint_strat)
                    new_potential = np.array([f(joint_strat) for f in potential_funcs])
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
        num_players = 2
        num_actions = 2
        potential_funcs = [self.custom_potential_wikipedia() for _ in range(self.batch_size)]
        payoff_matrices = potential(num_players,
                                    num_actions,
                                    potential_funcs,
                                    batch_size=self.batch_size,
                                    min_r=self.min_r,
                                    max_r=self.max_r,
                                    seed=self.seed)

        self.check_potential(payoff_matrices, potential_funcs, np.ones((self.batch_size, 2)))

    def test_potential(self):
        num_players = 3
        num_actions = 5
        rng = np.random.default_rng(self.seed)
        potential_funcs = decreasing(num_players, self.batch_size, num_points=num_actions, seed=self.seed)
        weights = rng.uniform(low=0, high=1, size=(self.batch_size, num_players))
        payoff_matrices = potential(num_players,
                                    num_actions,
                                    potential_funcs,
                                    batch_size=self.batch_size,
                                    weights=weights,
                                    min_r=self.min_r,
                                    max_r=self.max_r,
                                    seed=self.seed)

        self.check_potential(payoff_matrices, potential_funcs, weights)

    def test_congestion_as_potential(self):
        num_players = 2
        num_facilities = 2
        payoff_funcs = decreasing(1, self.batch_size * num_facilities, num_points=num_players + 1, seed=self.seed)
        payoff_funcs = [payoff_funcs[i::self.batch_size] for i in range(self.batch_size)]
        payoff_matrices = congestion(num_players, num_facilities, payoff_funcs, batch_size=self.batch_size)
        weights = np.ones((self.batch_size, num_players))
        potential_funcs = [create_congestion_potential_func(funcs, num_facilities) for funcs in payoff_funcs]
        self.check_potential(payoff_matrices, potential_funcs, weights)


if __name__ == '__main__':
    unittest.main()
