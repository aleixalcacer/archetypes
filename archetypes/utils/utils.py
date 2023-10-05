import numpy as np


def check_generator(generator=None):
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
    if generator is None:
        generator = np.random.default_rng()
    elif isinstance(generator, int):
        generator = np.random.default_rng(generator)
    elif isinstance(generator, np.random.Generator):
        pass
    else:
        raise ValueError(f"generator must be an int, a Generator or None, got {generator}")

    return generator
