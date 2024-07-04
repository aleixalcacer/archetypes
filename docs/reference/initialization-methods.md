(initialization-methods)=

# Initialization Methods

Initialization methods are used to select the initial archetypes for the archetypal analysis.
The choice of initialization method may have a significant impact on the results.
The available initialization methods are:

## Uniform

The Uniform method is the simplest approach, initializing the archetypes by randomly selecting data points
uniformly across the dataset.

## Furthest First

The Furthest First {cite:p}`gonzalez_clustering_1985` algorithm starts by randomly selecting the first center or archetype.
It then iteratively adds the data point that is furthest from the nearest already selected center or archetype.

## Furthest Sum

Furthest Sum {cite:p}`morup_archetypal_2012` is a modification of the Furthest First method.
It sums the distances to all previously selected points to determine the next point.
For improved performance, the initial randomly chosen point is usually discarded and replaced by a new point
selected using the same criteria.

## AA++

AA++ {cite:p}`mair_archetypal_2024` starts by selecting the first archetype randomly at uniform.
The second archetype is chosen based on a distribution that assigns probabilities proportional to the distances
from the first archetype.
Subsequent archetypes are selected using a probability distribution where the likelihood of choosing a
point is proportional to the minimum distance to the convex hull of the already chosen archetypes.
