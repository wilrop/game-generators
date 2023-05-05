from battle_of_the_sexes import battle_of_the_sexes
from congestion import congestion
from covariant import covariant
from majority_voting import majority_voting
from random_uniform import random_uniform


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
    elif game == 'battle_of_the_sexes':
        batch = battle_of_the_sexes(batch_size=batch_size, **kwargs)
    else:
        raise ValueError(f'Unknown game {game}')

    return batch
