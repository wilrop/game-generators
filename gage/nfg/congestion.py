import numpy as np


def congestion(num_players, num_facilities, batch_size=1, payoff_funcs=None):
    """Create a congestion game.

    In the congestion game, each player chooses a subset from the set of all facilities. Each player then receives a
    payoff which is the sum of payoff functions for each facility in the chosen subset. Each payoff function depends
    only on the number of other players who have chosen the facility. Functions used with this generator should always
    be decreasing in order for the resulting game to meet the criteria for being considered a congestion game.

    Args:
        num_players (int): The number of players.
        num_facilities (int): The number of facilities.
        batch_size (int, optional): The batch size. Defaults to 1.
        payoff_funcs (list, optional): A list of payoff functions. Expects one payoff function per facility.
            Defaults to None.

    Returns:
        np.ndarray: A congestion game.
    """
    if payoff_funcs is None:
        payoff_funcs = [lambda x: 1 / (x + 1)] * num_facilities

    if num_facilities > 5:
        import warnings
        warnings.warn("The number of actions is 2 to the power of the number of facilities. The game matrix grows "
                      "rapidly at a large numbers of facilities!")

    from itertools import chain, combinations, product
    subsets = list(chain.from_iterable(combinations(range(num_facilities), r) for r in range(1, num_facilities + 1)))
    joint_subsets = product(*[subsets] * num_players)
    num_actions = len(subsets)
    action_shape = (*[num_actions] * num_players,)
    payoff_matrices = np.zeros((batch_size, num_players, *action_shape))

    for joint_subset, idx in zip(joint_subsets, np.ndindex(*action_shape)):
        players_at_facilities = np.bincount(np.concatenate(joint_subset))
        for player, player_subset in enumerate(joint_subset):
            payoffs = sum(payoff_funcs[i](players_at_facilities[i], batch_size=batch_size) for i in player_subset)
            payoff_matrices[(slice(None), player) + idx] = payoffs

    if batch_size == 1:
        return payoff_matrices[0]
    else:
        return payoff_matrices
