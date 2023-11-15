import numpy as np
from itertools import product
from gage.functions.interpolate import interpolate_table
from gage.utils.transforms import scale_array

def concave_table(dim, batch_size=1, min_y=0, max_y=10, num_points=10, rng=None, seed=None):
    """Generate a random concave, monotonically increasing, function as a table of points.

    Note:
        The function is generated by constructing a piecewise linear function where the gradient of each piece is
        decreasing.

    Note:
        This function can be made continuous by using linear interpolation between the points.

    Args:
        dim (int): The dimension of the function.
        batch_size (int, optional): The batch size. Defaults to 1.
        min_y (int, optional): The minimum y value. Defaults to 0.
        max_y (int, optional): The maximum y value. Defaults to 10.
        num_points (int, optional): The number of points. Defaults to 10.
        rng (np.random.Generator, optional): The random number generator. Defaults to None.
        seed (int, optional): The random seed. Defaults to None.

    Returns:
        np.ndarray: A random concave function as a table.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    max_deriv = max_y - min_y
    concave_table = np.zeros(shape=(batch_size, *([num_points] * dim)))
    concave_table[(slice(None),) + ((0,) * dim)] = max_deriv
    concave_table[(slice(None),) + ((-1,) * dim)] = 1e-6  # Set final derivative to a low value.

    inserted = np.array([(0,) * dim, (num_points - 1,) * dim])  # Keep track of which points have been inserted.
    to_insert = list(product(*[range(num_points) for _ in range(dim)]))[1:-1]  # Generate all points to insert.
    rng.shuffle(to_insert)  # Shuffle the points to avoid bias from the uniform distribution.

    # Sort points along each dimension
    for idx in to_insert:
        # Generate the constraining indices. These are the indices which are one below in each dimension.
        lower_constraining_idxs = inserted[np.all(inserted <= idx, axis=-1)]
        upper_constraining_idxs = inserted[np.all(inserted >= idx, axis=-1)]

        # Extract the constraining values.
        lower_constraining = []
        upper_constraining = []
        for lower_constraining_idx in lower_constraining_idxs:
            lower_constraining.append(concave_table[(slice(None),) + tuple(lower_constraining_idx)])
        for upper_constraining_idx in upper_constraining_idxs:
            upper_constraining.append(concave_table[(slice(None),) + tuple(upper_constraining_idx)])

        # Compute the bounds for the new derivative.
        max_deriv = np.min(lower_constraining, axis=0)
        min_deriv = np.max(upper_constraining, axis=0)

        # Generate a new derivative which respects the bounds.
        concave_table[(slice(None),) + idx] = rng.uniform(low=min_deriv, high=max_deriv)
        inserted = np.vstack((inserted, idx))

    for d in range(dim):
        concave_table = concave_table.cumsum(axis=d + 1)

    # Extract the minimum and maximum values for each batch.
    min_values = concave_table[(slice(None),) + ((0,) * dim)].reshape(batch_size, *([1] * dim))
    max_values = concave_table[(slice(None),) + ((-1,) * dim)].reshape(batch_size, *([1] * dim))

    # Scale the table to the desired range
    return scale_array(concave_table, min_values, max_values, min_y, max_y)


def concave(dim, batch_size=1, batched=True, min_y=0, max_y=10, num_points=10, rng=None, seed=None):
    """Generate a random concave, monotonically increasing, function.

    Args:
        dim (int): The dimension of the function.
        batch_size (int, optional): The batch size. Defaults to 1.
        batched (bool, optional): Whether to return a batched function. Defaults to True.
        min_y (int, optional): The minimum y value. Defaults to 0.
        max_y (int, optional): The maximum y value. Defaults to 10.
        num_points (int, optional): The number of points. Defaults to 10.
        rng (np.random.Generator, optional): The random number generator. Defaults to None.
        seed (int, optional): The random seed. Defaults to None.

    Returns:
        list | callable: A collection of of concave functions or a single concave function.
    """
    concave_t = concave_table(dim, batch_size, min_y, max_y, num_points, rng, seed)
    batched = batch_size > 1 or batched
    if not batched:
        concave_t = concave_t[0]
    concave_f = interpolate_table(concave_t, batched=batched, method='linear')
    return concave_f
