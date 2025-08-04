from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x, y = make_blobs(n_samples=300, centers=3, random_state=42)

model = KMeans(n_clusters=3, random_state=42)
model.fit(x)

centers = model.cluster_centers_

plt.scatter(x[:, 0], x[:, 1], c=model.labels_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200)
plt.show()
