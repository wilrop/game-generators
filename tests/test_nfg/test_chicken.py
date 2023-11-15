import unittest
from gage.nfg.chicken import chicken


class TestChicken(unittest.TestCase):
    batch_size = 100
    min_r = 10
    max_r = 20
    seed = 0

    def test_chicken(self):
        payoff_matrices = chicken(batch_size=self.batch_size, min_r=self.min_r, max_r=self.max_r, seed=self.seed)

        for payoffs in payoff_matrices:
            m1, m2 = payoffs
            b = m1[0, 0]
            c = m1[0, 1]
            a = m1[1, 0]
            d = m1[1, 1]
            self.assertTrue(b == m2[0, 0])
            self.assertTrue(a == m2[0, 1])
            self.assertTrue(c == m2[1, 0])
            self.assertTrue(d == m2[1, 1])
            self.assertTrue(a > b > c > d)


if __name__ == "__main__":
    unittest.main()
