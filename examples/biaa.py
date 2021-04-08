import archetypes as arch
import numpy as np
import matplotlib.pyplot as plt

shape = (200, 20)
random_state = np.random.RandomState(0)

# Generate some random data
X = random_state.normal(0, 1, shape)
plt.plot(X[:, 0], X[:, 1], '.')

# Create an AA estimator and fit it
biaa = arch.BiAA(k=2, c=2, n_init=5, max_iter=1000, random_state=random_state)
biaa.fit(X)

print(biaa.archetypes_)