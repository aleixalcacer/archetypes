# Package Overview

The Archetypes package provides algorithms for performing Archetypal Analysis (AA) on various data types (NumPy arrays, Pandas dataframes, PyTorch tensors) and includes visualization tools for the results.


# What is Archetypal Analysis?

Archetypal Analysis identifies a small number of archetypes, which are extreme examples of a dataset. These archetypes help in dimensionality reduction, outlier detection, and clustering by representing the most extreme points in the data.


# Benefits of Archetypal Analysis

**Dimensionality Reduction:** Reduces data dimensionality by identifying key archetypes.

**Interpretability:** Provides insights into data patterns through characteristic examples.

**Outlier Detection:** Identifies observations that cannot be well represented by archetypes.

**Clustering:** Partitions data into groups represented by different subsets of archetypes.


# Initialization Methods

## AA (Archetypal Analysis):

**algorithm_init="random":** Initializes archetypes randomly.

**algorithm_init="furthest_sum":** Uses the furthest sum method for better spread.

### Parameters:

**n_archetypes:** Number of archetypes.

**n_init:** Number of initialization.

**max_iter:** Maximum iterations.

**verbose:** Verbosity flag.

**tol:** Tolerance for convergence.

**random_state:* Seed for random number generation.

## BiAA (Biarchetypal Analysis):

### Parameters: 

Similar structure to AA with specific differences for biarchetypal analysis.

## NAA (N-archetypal Analysis)

### Parameters:

Similar structure to AA but adapted for N-archetypal analysis.

## furthest_sum:

**Description:** Function to initialize points using the furthest sum method.


# Additional Functions

## make_archetypal_dataset

**Description:** Creates an archetypal dataset.

## permute_dataset

**Description:** Permutes the dataset.

## shuffle_dataset

**Description:** Shuffles the dataset.

## sort_by_archetype_similarity

**Description:** Sorts data by archetype similarity.

## sort_by_labels

**Description:** Sorts data by labels.

## check_generator

Description:

**Description:** Utility function to check the validity of a generator.


# Visualization Functions

## simplex

**Description:** Generates a simplex visualization.

### Usage Example:

simplex(data)

## bisimplex

**Description:** Generates a bisimplex visualization.

### Usage Example:

bisimplex(data)

## heatmap

**Description:** Generates a heatmap visualization.

### Usage Example:

heatmap(data)


# Comparison of Methods

* **Random Initialization:** Best for general use, quick setup.
* **Furthest Sum Initialization:** Useful when a better spread of initial points is needed.
* **BiAA:** Used specifically for biarchetypal analysis scenarios.
* **NAA:** Suitable for N-archetypal analysis.
* **make_archetypal_dataset:** Helpful for creating datasets tailored to archetypal analysis.
* **permute_dataset, shuffle_dataset:** Useful for data manipulation and preparation.
* **sort_by_archetype_similarity, sort_by_labels:** Assist in organizing and sorting data based on specific criteria.
* **Visualization Functions:** Useful for visualizing data and results in different formats.