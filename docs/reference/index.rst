API Reference
=============

.. currentmodule:: archetypes

Here, you will find detailed documentation of all the functions, classes, and methods available in our API.

Whether you are a developer who is new to our API or an experienced user looking for more advanced features, we hope you find this documentation helpful. If you have any questions or feedback, please don't hesitate to reach out to us.

Algorithms
----------

This section provides an overview of the main algorithms implemented in the `archetypes` package. Each algorithm is designed to address different aspects of archetypal analysis, offering flexibility and performance across various computational backends. Explore the summaries and links below to learn more about their features, usage, and implementation details.

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
   SymmetricBiAA
   NAA


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

This section introduces dataset-related utilities included in the `archetypes` package. These functions allow you to generate synthetic datasets with known archetypal structures, load benchmark datasets, and create controlled experimental scenarios. They are essential for algorithm validation, performance benchmarking, testing new methodological ideas, and educational purposes in archetypal analysis.

.. currentmodule:: archetypes.datasets

.. autosummary::
   :toctree: _autosummary/datasets
   :recursive:
   :template: function.rst

   make_archetypal_dataset


Processing
----------

This section covers data processing and manipulation utilities designed specifically for archetypal analysis workflows. These functions help you organize, sort, and transform your data and analysis results to extract meaningful insights. Use these tools to prepare datasets for analysis, post-process archetypal results, find representative samples, and create ordered visualizations that reveal archetypal patterns and relationships in your data.

.. currentmodule:: archetypes.processing

.. autosummary::
   :toctree: _autosummary/processing
   :recursive:
   :template: function.rst

   permute
   shuffle
   sort_by_coefficients
   sort_by_labels
   get_closest_n
   get_closest_threshold


.. toctree:: _autosummary
   :caption: More information
   :maxdepth: 1
   :hidden:

   initialization-methods
   optimization-methods
   references
