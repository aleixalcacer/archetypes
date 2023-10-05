import numpy as np

from archetypes.datasets import make_archetypal_dataset

random_state = False

n_archetypes = (2, 2)
shape = (10, 10)

data = make_archetypal_dataset(
    archetypes=np.array([[0.75, 0.1], [0.0, 0.9]]), shape=shape, alpha=1, random_state=random_state
)

# data = make_checkerboard(
#     shape=shape,
#     n_clusters=n_archetypes,
#     noise=0.1,
#     minval=0,
#     maxval=1,
#     random_state=random_state,
# )
#
#
# import torch
# data = torch.from_numpy(data).float()
#
# mod = arch.torch.NAA(k=n_archetypes, s=data.shape, device="cpu")
# mod.train(data, n_epochs=2_000, learning_rate=0.05)
# print(mod.Z)
