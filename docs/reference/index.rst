API Reference
=============

.. currentmodule:: archetypes

Here, you will find detailed documentation of all the functions, classes, and methods available in our API.

Whether you are a developer who is new to our API or an experienced user looking for more advanced features, we hope you find this documentation helpful. If you have any questions or feedback, please don't hesitate to reach out to us.

Algorithms
----------

This section provides an overview of the main algorithms implemented in the `archetypes` package. Each algorithm is designed to address different aspects of archetypal analysis, offering flexibility and performance across various computational backends. Explore the summaries and links below to learn more about their features, usage, and implementation details.

Numpy backend
~~~~~~~~~~~~~

.. currentmodule:: archetypes
.. autosummary::
   :caption: API Reference
   :toctree: _autosummary/numpy
   :recursive:
   :template: class.rst

   AA
   KernelAA
   FairAA
   FairKernelAA
   ADA
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

This section covers the visualization tools provided by the `archetypes` package. These utilities help you interpret and present the results of archetypal analysis through a variety of plots and graphical representations. Use these functions to gain insights into your data and effectively communicate your findings.

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

This section introduces dataset-related utilities included in the `archetypes` package. These functions allow you to generate synthetic datasets, manipulate existing data, and organize samples for archetypal analysis. They are useful for benchmarking algorithms, testing new ideas, and preparing data for further exploration.

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
   references
