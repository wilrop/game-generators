import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interpolate_table(table_function, batched=False, method='linear'):
    """Interpolate a table function.

    Note:
        This function is a wrapper for scipy.interpolate.RegularGridInterpolator.

    Args:
        table_function (np.array): A table function.
        batched (bool, optional): Whether to return a batched interpolation function. Defaults to False.
        method (str, optional): The interpolation method. Defaults to 'linear'.

    Returns:

    """
    table_shape = table_function.shape if not batched else table_function.shape[1:]
    table_points = [np.arange(n) for n in table_shape]

    if not batched:
        return RegularGridInterpolator(table_points, table_function, method=method)
    else:
        batch_size = table_function.shape[0]

    interpolators = [RegularGridInterpolator(table_points, table_function[i], method=method) for i in range(batch_size)]

    def batched_interp(points):
        points = np.array(points)
        pshape = points.shape
        points = points.reshape(batch_size, -1, len(table_shape))  # Flatten into points of shape (Batch, N, dim).
        res = np.array([interpolator(p) for interpolator, p in zip(interpolators, points)])
        final_shape = pshape[:-1]
        return res.reshape(final_shape)

    return batched_interp
