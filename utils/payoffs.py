import numpy as np


def strictly_decreasing(num, batch_size=1, min_r=0, max_r=5, min_diff=1e-4, rng=None, seed=None):
    """Generates a strictly decreasing sequence of arrays.

    Args:
        num (int): The number of arrays to generate.
        batch_size (int, optional): The batch size. Defaults to 1.
        min_r (int, optional): The minimum reward. Defaults to 0.
        max_r (int, optional): The maximum reward. Defaults to 5.
        min_diff (float, optional): The minimum difference between the largest and smallest elements. Defaults to 1e-4.
        rng (np.random.Generator, optional): The random number generator. Defaults to None.
        seed (int, optional): The random seed. Defaults to None.

    Returns:
        list: A list of arrays.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)

    res = [max_r]
    for i in range(num):
        res.append(rng.uniform(low=min_r + (num - i - 1) * min_diff, high=res[-1], size=batch_size))
    return res[1:]
