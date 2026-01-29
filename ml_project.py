# Machine Learning Project
# Live Class Assignment

import numpy as np
from sklearn.cluster import KMeans

# Sample dataset
X = np.array([[30], [35], [80], [85]])

# K-Means model
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# Output
print("Cluster labels:", kmeans.labels_) 
