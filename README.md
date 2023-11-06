# GAGE

Game Generators is a package for efficiently generating games for research and experimentation. It is loosely based
on [GAMUT](http://gamut.stanford.edu/). Its main features are:

1. Efficiently generate batches of games drawn from a specific distribution that can be used for deep learning purposes.
2. Wide range of available game structures.
3. In-depth documentation that provides references to the literature.

## Format

Every game is returned with separate payoff tensors per player. For example, when generating a batch of 1O games with 2
players and 2 actions per player, we return a (10, 2, 2, 2) tensor where the first dimension contains the batch, the
second dimension contains a payoff tensor for an individual player, and the remainder is the joint action actions.

It is common in game theoretic research to present joint payoff tensors. We provide a utility function to convert the
separate payoff tensors to a joint payoff tensor. This function is called `to_joint_payoff` and is available in
the `utils` module.

## Installation

For now, GAGE is only available for installation from source. To install, clone the repository and pip install the
necessary packages.

## Usage

GAGE is designed to be regular Python package. To use it, simply import the package and use the game generator that you
desire.

## Contributing

We are building a suite of game generators that can be used in modern game theoretic research. If you are working in
this area and want to get involved, contributions are very welcome! For major changes, please open an issue first to
discuss what you would like to change.

## Citation

If you use GAGE in your research, please use the following BibTeX entry:

```
@misc{ropke2023gage,
  author = {Willem Röpke},
  title = {GAGE: Game Generators},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/wilrop/mo-game-theory}},
}
```

## License

This project is licensed under the terms of the MIT license.