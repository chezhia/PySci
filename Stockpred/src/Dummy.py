import numpy as np
import matplotlib.pyplot as plt
#x = np.arange(0, 5, 0.1);
#y = np.sin(x)
#plt.plot(x, y)
#plt.show()
print('terminating...')
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');
print(len(X))
print (type(X))
plt.show()