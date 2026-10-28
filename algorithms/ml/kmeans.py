import numpy as np

def kmeans_steps(x,n_clusters=3,max_iter=10):

    idx=np.random.choice(len(x),n_clusters,replace=False)
    centroids=x[idx]
    steps=[]
    for i in range(max_iter):
        distance=np.linalg.norm(x[ : , np.newaxis]-centroids,axis=2)
        labels=np.argmin(distance,axis=1)

        steps.append((centroids.copy(), labels.copy()))

        new_centroids = np.array([x[labels == k].mean(axis=0) for k in range(n_clusters)])
        centroids = new_centroids
    return steps