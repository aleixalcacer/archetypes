from time import time

import numpy as np

from archetypes import AA

data = np.random.normal(0, 4, 2_000).reshape(-1, 2)

aa_kwargs = {
    "n_archetypes": 4,
    "max_iter": 200,
    "verbose": False,
    "tol": 1e-4,
}

mod0 = AA(**aa_kwargs, init="uniform")
t0 = time()
mod0.fit(data)
t1 = time()
print(f"mod0: {t1 - t0:.4f} s | RSS: {mod0.loss_:.2f}")

mod1 = AA(**aa_kwargs, init="furthest_sum")
t0 = time()
mod1.fit(data)
t1 = time()
print(f"mod1: {t1 - t0:.4f} s | RSS: {mod1.loss_:.2f}")
