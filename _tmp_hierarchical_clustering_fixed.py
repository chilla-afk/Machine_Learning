import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Import dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dendrogram
plt.figure(figsize=(8, 4))
linkage_matrix = sch.linkage(X_scaled, method='ward')
sch.dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.tight_layout()
plt.show()

# Agglomerative clustering (sklearn compatibility)
try:
    hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
except TypeError:
    hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

y_hc = hc.fit_predict(X_scaled)

# Visualize clusters (generic)
plt.figure(figsize=(7, 5))
colors = ['red','blue','green','cyan','magenta','orange','brown','purple','olive','teal']
for i, cluster_id in enumerate(np.unique(y_hc)):
    mask = y_hc == cluster_id
    plt.scatter(X[mask, 0], X[mask, 1], s=80, c=colors[i % len(colors)], label=f'Cluster {cluster_id}')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.tight_layout()
plt.show()

