import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interpolate_table(table_function, min_vec=None, max_vec=None, batched=False, method="linear"):
    """Interpolate a table function.

    Note:
        This function is a wrapper for scipy.interpolate.RegularGridInterpolator.

    Args:
        table_function (np.array): A table function.
        min_vec (np.array, optional): The minimum input values. Defaults to None.
        max_vec (np.array, optional): The maximum input values. Defaults to None.
        batched (bool, optional): Whether to return a batched interpolation function. Defaults to False.
        method (str, optional): The interpolation method. Defaults to 'linear'.

    Returns:

    """
    if min_vec is None or max_vec is None:
        table_shape = table_function.shape if not batched else table_function.shape[1:]
        table_points = [np.arange(n) for n in table_shape]
    else:
        num_points = table_function.shape[-1]
        table_points = [np.linspace(min_v, max_v, num_points) for min_v, max_v in zip(min_vec, max_vec)]

    if not batched:
        return RegularGridInterpolator(table_points, table_function, method=method)
    else:
        return [RegularGridInterpolator(table_points, table, method=method) for table in table_function]
