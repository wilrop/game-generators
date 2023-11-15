import unittest
from gage.nfg.bach_stravinsky import bach_stravinsky


class TestIdentity(unittest.TestCase):
    batch_size = 100
    min_r = 10
    max_r = 20
    seed = 0

    def test_bach_stravinsky(self):
        payoff_matrices = bach_stravinsky(batch_size=self.batch_size,
                                          min_r=self.min_r,
                                          max_r=self.max_r,
                                          seed=self.seed)
        for payoffs in payoff_matrices:
            m1, m2 = payoffs
            if m1[0, 0] == m2[0, 0]:
                self.assertTrue(m1[0, 0] == m1[1, 1] == m2[1, 1])
                c = m1[0, 0]
                p1 = m1[0, 1]
                p2 = m1[1, 0]
                p3 = m2[0, 1]
                p4 = m2[1, 0]
            else:
                self.assertTrue(m1[0, 1] == m2[0, 1] == m1[1, 0] == m2[1, 0])
                c = m1[0, 1]
                p1 = m1[0, 0]
                p2 = m1[1, 1]
                p3 = m2[0, 0]
                p4 = m2[1, 1]

            self.assertTrue(p1 == p4 and p2 == p3)
            self.assertTrue(c < p1 < p2 or c < p2 < p1)


if __name__ == '__main__':
    unittest.main()
