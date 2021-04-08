import archetypes as arch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_checkerboard

random_state = np.random.RandomState(0)

n_clusters = (2, 2)
data, rows, columns = make_checkerboard(shape=(200, 200),
                                        n_clusters=n_clusters,
                                        shuffle=False,
                                        noise=10,
                                        random_state=random_state)

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")


model = arch.BiAA(n_archetypes=(2, 2), random_state=0)
model.fit(data)


plt.show()
plt.close()
