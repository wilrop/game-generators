import unittest
import numpy as np
from game_generators.nfg.war_of_attrition import war_of_attrition


class TestWarOfAttrition(unittest.TestCase):
    batch_size = 100
    min_r = 10
    max_r = 20
    min_dec = 1
    max_dec = 3
    seed = 0

    def check_war_of_attrition(self, timesteps, payoff_matrices):
        for payoffs in payoff_matrices:
            m1, m2 = payoffs
            v1 = m1[1, 0]
            v2 = m2[0, 1]
            dec1 = v1 - m1[2, 1]
            dec2 = v2 - m2[1, 2]

            for idx in np.ndindex(timesteps, timesteps):
                a1, a2 = idx
                if a1 == a2:
                    np.testing.assert_almost_equal(m1[a1, a2], (v1 / 2) - a1 * dec1)
                    np.testing.assert_almost_equal(m2[a1, a2], (v2 / 2) - a2 * dec2)
                elif a1 > a2:
                    np.testing.assert_almost_equal(m1[a1, a2], v1 - a2 * dec1)
                    np.testing.assert_almost_equal(m2[a1, a2], 0 - a2 * dec2)
                else:
                    np.testing.assert_almost_equal(m1[a1, a2], 0 - a1 * dec1)
                    np.testing.assert_almost_equal(m2[a1, a2], v2 - a1 * dec2)

    def test_small(self):
        timesteps = 5
        payoff_matrices = war_of_attrition(
            timesteps,
            batch_size=self.batch_size,
            min_r=self.min_r,
            max_r=self.max_r,
            min_dec=self.min_dec,
            max_dec=self.max_dec,
            seed=self.seed,
        )
        self.check_war_of_attrition(timesteps, payoff_matrices)

    def test_large(self):
        timesteps = 20
        payoff_matrices = war_of_attrition(
            timesteps,
            batch_size=self.batch_size,
            min_r=self.min_r,
            max_r=self.max_r,
            min_dec=self.min_dec,
            max_dec=self.max_dec,
            seed=self.seed,
        )
        self.check_war_of_attrition(timesteps, payoff_matrices)


if __name__ == "__main__":
    unittest.main()
