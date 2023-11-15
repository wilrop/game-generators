import unittest
from game_generators.nfg.grab_the_dollar import grab_the_dollar


class TestGrabTheDollar(unittest.TestCase):
    batch_size = 100
    min_r = 0
    max_r = 20
    seed = 0

    def check_grab_the_dollar(self, timesteps, payoff_matrices):
        for payoffs in payoff_matrices:
            m1, m2 = payoffs
            a1 = m1[0, 1]
            b1 = m1[1, 0]
            c1 = m1[0, 0]
            a2 = m2[1, 0]
            b2 = m2[0, 1]
            c2 = m2[0, 0]

            self.assertTrue(a1 > b1 > c1)
            self.assertEqual(a1, a2)
            self.assertEqual(b1, b2)
            self.assertEqual(c1, c2)

            for i in range(timesteps):
                for j in range(timesteps):
                    if i == j:
                        self.assertEqual(m1[i, j], c1)
                        self.assertEqual(m2[i, j], c2)
                    elif i < j:
                        self.assertEqual(m1[i, j], a1)
                        self.assertEqual(m2[i, j], b2)
                    else:
                        self.assertEqual(m1[i, j], b1)
                        self.assertEqual(m2[i, j], a2)

    def test_small(self):
        timesteps = 5
        payoff_matrices = grab_the_dollar(
            timesteps, batch_size=self.batch_size, min_r=self.min_r, max_r=self.max_r, seed=self.seed
        )
        self.check_grab_the_dollar(timesteps, payoff_matrices)

    def test_large(self):
        timesteps = 10
        payoff_matrices = grab_the_dollar(
            timesteps, batch_size=self.batch_size, min_r=self.min_r, max_r=self.max_r, seed=self.seed
        )
        self.check_grab_the_dollar(timesteps, payoff_matrices)


if __name__ == "__main__":
    unittest.main()
