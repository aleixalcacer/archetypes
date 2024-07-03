import numbers

import jax
import numpy as np
import torch


def check_generator_numpy(seed=None):
    """Check if generator is an int, a Generator or None and return a Generator.

    Parameters
    ----------
    generator : int, Generator or None
        The generator to check. If None, the default generator is used.

    Returns
    -------
    generator : Generator
        The checked generator.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState instance" % seed)


def check_generator_jax(seed=None):
    """Check seed and return a key for random number generation.

    Parameters
    ----------
    seed : int or None
        The generator to check. If None, 0 is used.

    Returns
    -------
    key : KeyArray
        The key to use for random number generation.
    """
    if seed is None:
        return jax.random.key(0)
    if isinstance(seed, int):
        return jax.random.key(seed)
    raise ValueError("%r cannot be used to seed a jax.random.KeyArray instance" % seed)


def check_generator_torch(generator=None, device=None):
    """Check if generator is an int, a Generator or None and return a Generator.

    Parameters
    ----------
    generator : int, Generator or None
        The generator to check. If None, the default generator is used.
    device : str or torch.device
        The device to use for the generator. If None, the default device is used.

    Returns
    -------
    generator : Generator
        The checked generator.
    """

    if device is None:
        device = "cpu"
    if generator is None:
        return torch.Generator(device=device)
    if isinstance(generator, numbers.Integral):
        return torch.Generator(device=device).manual_seed(generator)
    if isinstance(generator, torch.Generator):
        return generator
    raise ValueError("%r cannot be used to seed a torch.Generator instance" % generator)
