import numpy as np


def chicken(batch_size=1, min_r=0, max_r=5, rng=None, seed=None):
    """Creates a 2x2 game of chicken.

    This generates a matrix with payoffs

    b, b c, a
    a, c d, d

    Where a > b > c > d

    Generating for more than 2 players is under n-player chicken.

    Args:
        batch_size (int, optional): The batch size. Defaults to 1.
        min_r (int, optional): The minimum reward. Defaults to 0.
        max_r (int, optional): The maximum reward. Defaults to 5.
        rng (np.random.Generator, optional): The random number generator. Defaults to None.
        seed (int, optional): The random seed. Defaults to None.

    Returns:
        ndarray: A game of chicken.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    min_diff = 1e-4

    # Sequentially generate the a, b, c and d payoffs. The minimum payoffs for the largest payoffs are raised to ensure
    # a strictly decreasing sequence.
    a = rng.uniform(low=min_r + 3 * min_diff, high=max_r, size=batch_size)
    b = rng.uniform(low=min_r + 2 * min_diff, high=a, size=batch_size)
    c = rng.uniform(low=min_r + min_diff, high=b, size=batch_size)
    d = rng.uniform(low=min_r, high=c, size=batch_size)

    payoff_matrices = np.zeros((batch_size, 2, 2, 2))
    payoff_matrices[:, [0, 1], [1, 0], [0, 1]] = a.reshape(batch_size, -1)
    payoff_matrices[:, [0, 1], [0, 0], [0, 0]] = b.reshape(batch_size, -1)
    payoff_matrices[:, [0, 1], [0, 1], [1, 0]] = c.reshape(batch_size, -1)
    payoff_matrices[:, [0, 1], [1, 1], [1, 1]] = d.reshape(batch_size, -1)

    if batch_size == 1:
        return payoff_matrices[0]
    else:
        return payoff_matrices
