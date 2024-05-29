API Reference
=============

.. currentmodule:: archetypes

In this section, you will find detailed documentation of all the functions, classes, and methods available in our API.

Whether you are a developer who is new to our API or an experienced user looking for more advanced features, we hope you find this documentation helpful. If you have any questions or feedback, please don't hesitate to reach out to us.

Algorithms
----------

Scikit-learn
~~~~~~~~~~~~

.. currentmodule:: archetypes.sklearn
.. autosummary::
   :caption: API Reference
   :toctree: _autosummary/sklearn
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


.. toctree:: _autosummary
   :caption: More information
   :maxdepth: 1
   :hidden:

   optimization-methods
