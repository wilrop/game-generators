from gage.nfg.bach_stravinsky import bach_stravinsky
from gage.nfg.congestion import congestion
from gage.nfg.covariant import covariant
from gage.nfg.majority_voting import majority_voting
from gage.nfg.random_uniform import random_uniform


def generate_nfg(game, num_players, num_actions, batch_size=1, **kwargs):
    """Generate a batch of normal-form games.

    Args:
        game (str): The game to generate.
        num_players (int): The number of players.
        num_actions (int): The number of actions.
        batch_size (int, optional): The batch size. Defaults to 1.

    Returns:
        np.ndarray: The batch of games.
    """
    if game == 'majority-voting':
        batch = majority_voting(num_players, num_actions, batch_size=batch_size, **kwargs)
    elif game == 'random-uniform':
        batch = random_uniform(num_players, num_actions, batch_size=batch_size, **kwargs)
    elif game == 'covariant':
        batch = covariant(num_players, num_actions, batch_size=batch_size, **kwargs)
    elif game == 'congestion':
        batch = congestion(num_players, num_actions, batch_size=batch_size, **kwargs)
    elif game == 'bach_stravinsky':
        batch = bach_stravinsky(batch_size=batch_size, **kwargs)
    else:
        raise ValueError(f'Unknown game {game}')

    return batch
