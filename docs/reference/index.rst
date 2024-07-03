API Reference
=============

.. currentmodule:: archetypes

In this section, you will find detailed documentation of all the functions, classes, and methods available in our API.

Whether you are a developer who is new to our API or an experienced user looking for more advanced features, we hope you find this documentation helpful. If you have any questions or feedback, please don't hesitate to reach out to us.

Algorithms
----------

Numpy backend
~~~~~~~~~~~~~

.. currentmodule:: archetypes
.. autosummary::
   :caption: API Reference
   :toctree: _autosummary/numpy
   :recursive:
   :template: class.rst

   AA
   BiAA

JAX backend
~~~~~~~~~~~~~

.. currentmodule:: archetypes.jax
.. autosummary::
   :toctree: _autosummary/jax
   :recursive:
   :template: class.rst

   AA
   BiAA

Torch backend
~~~~~~~~~~~~~

.. currentmodule:: archetypes.torch
.. autosummary::
   :toctree: _autosummary/torch
   :recursive:
   :template: class.rst

   AA
   BiAA


Visualization
-------------

.. currentmodule:: archetypes.visualization

.. autosummary::
   :toctree: _autosummary/visualization
   :recursive:
   :template: function.rst

   simplex
   stacked_bar
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


.. toctree:: _autosummary
   :caption: More information
   :maxdepth: 1
   :hidden:

   initialization-methods
   optimization-methods
