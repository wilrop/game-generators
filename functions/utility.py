def polynomial(degree, min_coef, max_coef, rng=None, seed=None):
    """Generate a random polynomial of degree `degree` with coefficients drawn from a uniform distribution.

    Note:
        The coefficients are applied in reverse order due to the usage of the np.arange() function. This doesn't make a
        difference numerically but should be taken note of when applying the polynomial by hand.

    Args:
        degree (int): The degree of the polynomial.
        min_coef (float): The minimum coefficient.
        max_coef (float): The maximum coefficient.
        rng (np.random.Generator, optional): The random number generator. Defaults to None.
        seed (int, optional): The random seed. Defaults to None.

    Returns:
        np.ndarray: A random polynomial.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    coefficients = rng.uniform(low=min_coef, high=max_coef, size=degree + 1)
    poly_f = lambda x: np.dot(np.power(x, np.arange(degree + 1)), coefficients)
    return poly_f
