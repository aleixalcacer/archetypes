import archetypes as arch
import numpy as np
import matplotlib.pyplot as plt

shape = (200, 2)
random_state = np.random.RandomState(0)

# Generate some random data
X = random_state.normal(0, 1, shape)
plt.plot(X[:, 0], X[:, 1], '.')

# Create an AA estimator and fit it
aa = arch.AA(n_archetypes=5, n_init=5, max_iter=100, random_state=random_state)
aa.fit(X)

plt.plot(aa.archetypes_[:, 0], aa.archetypes_[:, 1], 'x')

# Define a point offside the hull convex of the archetypes
x = np.array([[-2.1, -2]])
plt.plot(x[:, 0], x[:, 1], 'o')

# Transform the point and recover it using the archetypes
x_trans = aa.transform(x)
x_new = x_trans @ aa.archetypes_
plt.plot(x_new[:, 0], x_new[:, 1], '*')

plt.show()
plt.close()
