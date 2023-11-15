from gage.nfg.bach_stravinsky import bach_stravinsky
from gage.nfg.bertrand_oligopoly import bertrand_oligopoly
from gage.nfg.chicken import chicken
from gage.nfg.congestion import congestion
from gage.nfg.covariant import covariant
from gage.nfg.discrete_uniform import discrete_uniform
from gage.nfg.grab_the_dollar import grab_the_dollar
from gage.nfg.hawk_dove import hawk_dove
from gage.nfg.majority_voting import majority_voting
from gage.nfg.potential import potential
from gage.nfg.random_uniform import random_uniform
from gage.nfg.war_of_attrition import war_of_attrition
from gage.nfg.zero_sum import zero_sum


def generate_nfg(game, *args, **kwargs):
    """Generate a batch of normal-form games from a specified distribution.

    Args:
        game (str): The game to generate.
        args: The arguments to pass to the generator.
        kwargs: The keyword arguments to pass to the generator.


    Returns:
        np.ndarray: The batch of games.
    """
    if game == "bach_stravinsky":
        return bach_stravinsky(*args, **kwargs)
    elif game == "bertrand_oligopoly":
        return bertrand_oligopoly(*args, **kwargs)
    elif game == "chicken":
        return chicken(*args, **kwargs)
    elif game == "congestion":
        return congestion(*args, **kwargs)
    elif game == "covariant":
        return covariant(*args, **kwargs)
    elif game == "discrete_uniform":
        return discrete_uniform(*args, **kwargs)
    elif game == "grab_the_dollar":
        return grab_the_dollar(*args, **kwargs)
    elif game == "hawk_dove":
        return hawk_dove(*args, **kwargs)
    elif game == "majority_voting":
        return majority_voting(*args, **kwargs)
    elif game == "potential":
        return potential(*args, **kwargs)
    elif game == "random_uniform":
        return random_uniform(*args, **kwargs)
    elif game == "war_of_attrition":
        return war_of_attrition(*args, **kwargs)
    elif game == "zero_sum":
        return zero_sum(*args, **kwargs)
    else:
        raise ValueError(f"Unknown game {game}")
