import numpy as np


def battle_of_the_sexes(batch_size=1, min_r=0, max_r=5, rng=None, seed=None):
    """Creates a 2x2 Battle of the Sexes Game.

    This generates a table:

    A, B C, C
    C, C B, A

    or

    C, C B, A
    A, B C, C

    such that either (C < A < B) or (C < B < A).

    Args:
        batch_size (int, optional): The batch size. Defaults to 1.
        min_r (int, optional): The minimum reward. Defaults to 0.
        max_r (int, optional): The maximum reward. Defaults to 5.
        rng (np.random.Generator, optional): The random number generator. Defaults to None.
        seed (int, optional): The random seed. Defaults to None.

    Returns:
        np.ndarray: A battle of the sexes game.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    min_diff = 1e-4

    # Sequentially generate the A, B and C payoffs. The minimum payoffs for the largest payoffs are raised to ensure
    # a strictly decreasing sequence.
    a = rng.uniform(low=min_r + 2 * min_diff, high=max_r, size=batch_size)
    b = rng.uniform(low=min_r + min_diff, high=a, size=batch_size)
    c = rng.uniform(low=min_r, high=b, size=batch_size)

    # Randomly swap entries from a and b to ensure that sometimes (C < A < B) and sometimes (C < B < A).
    swap = rng.choice([True, False], size=batch_size)
    a[swap], b[swap] = b[swap], a[swap]

    # Generates of the first form first.
    payoff_matrices = np.zeros((batch_size, 2, 2, 2))
    payoff_matrices[:, 0, 0, 0] = a
    payoff_matrices[:, 0, 1, 1] = b
    payoff_matrices[:, 1, 0, 0] = b
    payoff_matrices[:, 1, 1, 1] = a
    payoff_matrices[:, 0, 0, 1] = c
    payoff_matrices[:, 0, 1, 0] = c
    payoff_matrices[:, 1, 0, 1] = c
    payoff_matrices[:, 1, 1, 0] = c

    # Randomly swap the first and second rows.
    swap = rng.choice([True, False], size=batch_size)
    payoff_matrices[swap, :, 0, :], payoff_matrices[swap, :, 1, :] = payoff_matrices[swap, :, 1, :], payoff_matrices[
                                                                                                     swap, :, 0, :]

    if batch_size == 1:
        return payoff_matrices[0]
    else:
        return payoff_matrices
