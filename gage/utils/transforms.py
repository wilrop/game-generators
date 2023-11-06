import numpy as np


def to_joint_payoff(games, batched=True):
    """Convert a batch of games to joint payoff format.

    Args:
        games (np.ndarray): The batch of games.
        batched (bool, optional): Whether the games are batched or not. Defaults to True.

    Returns:
        np.ndarray: The batch of games in normal form.
    """
    return np.moveaxis(games, 0 + batched, -1)
