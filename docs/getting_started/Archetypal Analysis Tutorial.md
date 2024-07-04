# Archetypal Analysis Tutorial

This tutorial demonstrates how to perform Archetypal Analysis (AA) using the Archetypes package. AA identifies a small number of archetypes, which are extreme examples of a dataset.

## Step 1: Import Libraries and Load Data

### In [33]

First, import the necessary libraries and load the dataset.

<img width="310" alt="Screenshot 2024-07-04 at 5 15 57 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/6522d998-4b85-4276-a4bc-1b95382acbff">

* Load the dataset

<img width="151" alt="Screenshot 2024-07-04 at 5 16 33 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/0d5bdaf6-cd87-49e5-8d39-55b0d2c97c2b">

## Step 2: Initialize and Fit the AA Model

### In [35]

Initialize the AA model with specified parameters and fit it to the data.

<img width="204" alt="Screenshot 2024-07-04 at 5 17 51 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/f11f54fc-9789-455c-9a12-efc545ba0619">

* Define method-specific arguments

<img width="238" alt="Screenshot 2024-07-04 at 5 18 35 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/1fedb32b-0f59-446d-9feb-49a0a93f79fa">

* Define the number of archetypes

<img width="138" alt="Screenshot 2024-07-04 at 5 19 07 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/e2d9868d-533b-4c08-8458-72d8ad671093">

* Initialize and fit the AA model

<img width="758" alt="Screenshot 2024-07-04 at 5 21 50 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/d8817d7d-662e-41ce-96b3-3f77604fdff9">

### Out [35]

<img width="455" alt="Screenshot 2024-07-04 at 5 22 35 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/0ce46b15-408e-4ebc-97b6-e2a93ad984af">

## Step 3: Sort Data by Archetype Similarity and Permute Similarity Degrees

### In [36]

Sort the data by archetype similarity and permute the similarity degrees.

<img width="611" alt="Screenshot 2024-07-04 at 5 25 03 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/f7eabd18-6ad4-415b-a94d-cfc015e719fb">

* Sort the data by archetype similarity

<img width="751" alt="Screenshot 2024-07-04 at 5 25 45 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/9f8ecdf6-32a3-4f6b-9348-4a256dfaa39b">

* Permute the similarity degrees

<img width="695" alt="Screenshot 2024-07-04 at 5 26 14 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/3222add5-939c-4f7c-8433-266e6e4ee23f">

## Step 4: Visualize the Permuted Similarity Degrees

## In [37]

Visualize the permuted similarity degrees using simplex and stacked bar visualizations.

<img width="455" alt="Screenshot 2024-07-04 at 5 27 00 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/83b4949d-33c3-456e-98fe-304e6bfd2330">

**Visual**

<img width="808" alt="Screenshot 2024-07-04 at 5 28 03 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/0c26ead7-8052-41eb-ac88-0ba8df593541">

## Example 

Click the link to see the [Full Example](https://github.com/aleixalcacer/archetypes/blob/c8423656725ed89ccb299a21acbe336b47707574/docs/getting_started/examples/aa.ipynb).

By following this tutorial, you should be able to perform Archetypal Analysis using the Archetypes package, visualize the loaded data and the identified archetypes, and understand the results effectively. If you have any questions or encounter any issues, please don't hesitate to reach out to us. We hope you find this tutorial helpful in learning about AA and using the Archetypes package!
