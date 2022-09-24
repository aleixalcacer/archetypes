import importlib.metadata

from .algorithms import AA, BiAA, furthest_sum
from .visualization import simplex

__all__ = ["AA", "BiAA", "furthest_sum", "simplex"]

__version__ = importlib.metadata.version("archetypes")
