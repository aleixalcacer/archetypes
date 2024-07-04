# Biarchetypal Analysis Tutorial

This tutorial demonstrates how to perform Biarchetypal Analysis (BiAA) using the Archetypes package. BiAA is an extension of Archetypal Analysis (AA) that allows for the analysis of two distinct but related sets of archetypes.

## Step 1: Import Libraries and Generate Archetypes

### In [1]

First, import the necessary libraries and generate the archetypes for the dataset.

<img width="451" alt="Screenshot 2024-07-04 at 3 46 42 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/e897c303-de37-4e01-af9c-a2aef983820b">

* Define the number of archetypes

<img width="183" alt="Screenshot 2024-07-04 at 3 47 25 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/89718a2c-6083-47c4-85f0-aef143bb1242">

* Generate random archetypes

<img width="353" alt="Screenshot 2024-07-04 at 3 48 41 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/a4a4047e-8130-4aac-b838-7a6b94fedfcd">

## Step 2: Visualize Generated Archetypes

### In [2]

Visualize the generated archetypes to understand their distribution.

<img width="262" alt="Screenshot 2024-07-04 at 3 49 43 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/9ae2bbbc-cf05-4431-9031-180e07b37d9c">

* Visualize the generated archetypes

<img width="299" alt="Screenshot 2024-07-04 at 3 50 31 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/1d96f97b-3459-4c87-ad00-97720da63ec5">

**Visual**

<img width="446" alt="Screenshot 2024-07-04 at 3 52 09 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/14d3a3fd-59fb-4626-88d9-0feb2d0cade5">

## Step 3: Create an Archetypal Dataset

### In [3]

Generate the archetypal dataset using the generated archetypes.

<img width="649" alt="Screenshot 2024-07-04 at 3 53 56 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/38e4b91c-f868-4c12-9edc-06c881fc4b28">

## Step 4: Visualize the Generated Dataset

### In [4]

Visualize the archetypal dataset.

<img width="259" alt="Screenshot 2024-07-04 at 3 55 15 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/7c0960a5-c327-4706-b1c1-1817b05c4e5e">

* Visualize the generated dataset

<img width="229" alt="Screenshot 2024-07-04 at 3 56 04 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/b08426f0-8eff-40e4-bbad-2732c41ef86b">

**Visual**

<img width="441" alt="Screenshot 2024-07-04 at 3 58 49 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/d40c8028-df3b-471e-9913-5bd37bd665f8">

## Step 5: Initialize and Fit the BiAA Model

### In [5]

Initialize the BiAA model with specified parameters and fit it to the generated dataset.

<img width="262" alt="Screenshot 2024-07-04 at 3 59 22 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/a54fbec9-eee7-4b06-9a36-16a3a2eedd20">

* Define method-specific arguments

<img width="645" alt="Screenshot 2024-07-04 at 4 00 17 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/9c392081-0b93-4d35-a779-2785e3bb25c7">

* Initialize the BiAA model

<img width="267" alt="Screenshot 2024-07-04 at 4 01 00 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/58a09ed9-cce3-499f-b608-528a2fc36cd8">

* Fit the model to the dataset

<img width="104" alt="Screenshot 2024-07-04 at 4 01 37 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/790a91dc-4b51-4310-b54f-4ce013c02121">

### Out [5]

<img width="722" alt="Screenshot 2024-07-04 at 4 02 24 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/4314e3a1-cdcf-42ea-a62a-679ea13b7930">

## Step 6: Visualize the Learned Archetypes

### In [6]

Visualize the learned archetypes after fitting the model.

<img width="347" alt="Screenshot 2024-07-04 at 4 13 01 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/ec2e3ec5-c9fc-4616-b29e-c162eb2027ae">

**Visual**

<img width="455" alt="Screenshot 2024-07-04 at 4 06 10 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/b8a8d21e-3741-4579-be9e-992408eb019a">

## Step 7: Visualize the Similarity Degrees

### In [7]

Use the simplex visualization to plot the similarity degrees of the archetypes.

<img width="359" alt="Screenshot 2024-07-04 at 4 16 18 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/b295dce8-75dc-4a95-8dc4-c20a436b8985">

**Visual**

<img width="623" alt="Screenshot 2024-07-04 at 4 17 21 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/5fb2ced4-b2cb-4cb0-be53-810d3a600d9c">

## Step 8: Plot Model Loss

### In [8]

Plot the loss to visualize the convergence of the model.

<img width="173" alt="Screenshot 2024-07-04 at 4 23 44 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/a596a47d-5b5f-41ec-88b2-507c12842e02">

### Out [8]

<img width="349" alt="Screenshot 2024-07-04 at 4 24 33 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/5fc6e7be-c1ff-4ffc-9035-dabffb7b5e90">

**Visual**

<img width="558" alt="Screenshot 2024-07-04 at 4 25 11 PM" src="https://github.com/Whitegabriella789/archetypes/assets/172323441/8f7d3603-a120-49b4-9651-24e49e73a040">

## Example

Click the link to see the [Full Example](https://github.com/aleixalcacer/archetypes/blob/c8423656725ed89ccb299a21acbe336b47707574/docs/getting_started/examples/biaa.ipynb).

By following this tutorial, you should be able to perform Biarchetypal Analysis using the Archetypes package, visualize the generated archetypes, and understand the results effectively. If you have any questions or encounter any issues, please don't hesitate to reach out to us. We hope you find this tutorial helpful in learning about BiAA and using the Archetypes package!
