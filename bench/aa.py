from time import time

import numpy as np

import archetypes as arch

data = np.random.normal(0, 4, 2_000).reshape(-1, 2)

aa_kwargs = {
    "n_archetypes": 4,
    "n_init": 5,
    "max_iter": 200,
    "verbose": False,
    "tol": 1e-4,
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
