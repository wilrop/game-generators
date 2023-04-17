from polynomial import polynomial


def linear(min_coef, max_coef, rng=None, seed=None):
    """Generate a random linear function with coefficients drawn from a uniform distribution.

    Note:
        Generates a random polynomial of degree 1.

    Args:
        min_coef (float): The minimum coefficient.
        max_coef (float): The maximum coefficient.
        rng (np.random.Generator, optional): The random number generator. Defaults to None.
        seed (int, optional): The random seed. Defaults to None.

    Returns:
        callable: A random linear function.
    """
    return polynomial(1, min_coef, max_coef, rng, seed)
