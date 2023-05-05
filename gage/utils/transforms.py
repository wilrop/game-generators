import numpy as np


def normal_form(games, batched=True):
    """Convert a batch of games to normal form.

    Args:
        games (np.ndarray): The batch of games.
        batched (bool, optional): Whether the games are batched or not. Defaults to True.

    Returns:
        np.ndarray: The batch of games in normal form.
    """
    return np.moveaxis(games, 0 + batched, -1)
