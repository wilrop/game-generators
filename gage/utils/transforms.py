import numpy as np


def coordinate_grid(shape):
    """Create a coordinate grid.

    Args:
        shape (Tuple[int]): The shape of the grid.

    Returns:
        np.ndarray: The coordinate grid.
    """
    indices = np.expand_dims(np.indices(shape), axis=-1)
    return np.concatenate(indices, axis=-1)


def make_batched(arr, batch_size):
    """Make an array batched.

    Args:
        arr (array_like): The array to batch.
        batch_size (int): The batch size.

    Returns:
        np.ndarray: The batched array.
    """
    return np.repeat(np.expand_dims(arr, axis=0), batch_size, axis=0)


def to_joint_payoff(games, batched=True):
    """Convert a batch of games to joint payoff format.

    Args:
        games (np.ndarray): The batch of games.
        batched (bool, optional): Whether the games are batched or not. Defaults to True.

    Returns:
        np.ndarray: The batch of games in normal form.
    """
    return np.moveaxis(games, 0 + batched, -1)
