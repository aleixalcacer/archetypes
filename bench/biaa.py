from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_checkerboard

import archetypes as arch

random_state = np.random.RandomState(12)

n_archetypes = (3, 3)
data, rows, columns = make_checkerboard(
    shape=(90, 90),
    n_clusters=n_archetypes,
    shuffle=False,
    noise=4,
    random_state=random_state,
)
plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")
plt.show()

aa_kwargs = {
    "n_archetypes": n_archetypes,
    "n_init": 5,
    "max_iter": 10_000,
    "verbose": False,
    "tol": 1e-8,
}

mod0 = arch.AA(**aa_kwargs, algorithm_init="random")
t0 = time()
mod0.fit(data)
t1 = time()
print(f"mod0: {t1 - t0:.4f} s | RSS: {mod0.rss_:.2f}")

mod1 = arch.AA(**aa_kwargs, algorithm_init="furthest_sum")
t0 = time()
mod1.fit(data)
t1 = time()
print(f"mod1: {t1 - t0:.4f} s | RSS: {mod1.rss_:.2f}")

plt.matshow(mod0.archetypes_, cmap=plt.cm.Blues)
plt.title("Archetypal0 dataset")
plt.show()
plt.matshow(mod1.archetypes_, cmap=plt.cm.Blues)
plt.title("Archetypal1 dataset")
plt.show()
