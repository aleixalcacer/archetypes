API Reference
=============

.. currentmodule:: archetypes

In this section, you will find detailed documentation of all the functions, classes, and methods available in our API.

Whether you are a developer who is new to our API or an experienced user looking for more advanced features, we hope you find this documentation helpful. If you have any questions or feedback, please don't hesitate to reach out to us.

Algorithms
----------

Scikit-learn
~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :recursive:
   :template: class.rst

   AA
   BiAA


PyTorch
~~~~~~~

.. currentmodule:: archetypes.torch

.. autosummary::
   :toctree: _autosummary/torch
   :template: class-pytorch.rst

   AA
   BiAA
   NAA


Visualization
-------------

.. currentmodule:: archetypes.visualization

.. autosummary::
   :toctree: _autosummary/visualization
   :recursive:
   :template: function.rst

   simplex
   bisimplex
   heatmap


Datasets
--------

.. currentmodule:: archetypes.datasets

.. autosummary::
   :toctree: _autosummary/datasets
   :recursive:
   :template: function.rst

   make_archetypal_dataset
   permute_dataset
   shuffle_dataset
   sort_by_archetype_similarity
   sort_by_labels


Utils
-----

.. currentmodule:: archetypes.utils

..  autosummary::
    :toctree: _autosummary/utils
    :recursive:
    :template: function.rst

    check_generator
