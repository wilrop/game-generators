import numpy as np
from itertools import chain, combinations


def strictly_decreasing(num, batch_size=1, min_r=0, max_r=5, min_diff=1e-4, rng=None, seed=None):
    """Generates a strictly decreasing sequence of arrays.

    Args:
        num (int): The number of arrays to generate.
        batch_size (int, optional): The batch size. Defaults to 1.
        min_r (int, optional): The minimum reward. Defaults to 0.
        max_r (int, optional): The maximum reward. Defaults to 5.
        min_diff (float, optional): The minimum difference between the largest and smallest elements. Defaults to 1e-4.
        rng (np.random.Generator, optional): The random number generator. Defaults to None.
        seed (int, optional): The random seed. Defaults to None.

    Returns:
        list: A list of arrays.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)

    res = [max_r]
    for i in range(num):
        res.append(rng.uniform(low=min_r + (num - i - 1) * min_diff, high=res[-1], size=batch_size))
    return res[1:]


def create_congestion_potential_func(funcs, num_facilities):
    """Create the potential function for the exact potential game representing the congestion game.

    Note:
        See https://www.cs.ubc.ca/~kevinlb/teaching/cs532l%20-%202013-14/Lectures/Congestion%20Games.pdf for a
        derivation of the potential function.

    Args:
        funcs (list): The payoff functions for each facility.
        num_facilities (int): The number of facilities.

    Returns:
        callable: The potential function.
    """
    actions = list(chain.from_iterable(combinations(range(num_facilities), r) for r in range(num_facilities + 1)))
    actions = actions[1:]
    num_actions = len(actions)

    def potential_single_joint_action(joint_action):
        subset_counts = np.bincount(joint_action, minlength=num_actions)
        facility_counts = np.zeros(num_facilities, dtype=int)
        for subset, count in enumerate(subset_counts):
            for facility in actions[subset]:
                facility_counts[facility] += count
        total = 0
        for facility, count in enumerate(facility_counts):
            for num in range(1, count + 1):
                total += funcs[facility]([num])[0]
        return total

    def potential_func(joint_actions):
        joint_actions = np.array(joint_actions)
        if joint_actions.ndim == 1:
            return potential_single_joint_action(joint_actions)
        else:
            return np.apply_along_axis(potential_single_joint_action, -1, joint_actions)

    return potential_func
