import numpy as np


def knn_steps(X, y, test_point, k=3):
    steps=[]

    distances =np.linalg.norm(X-test_point,axis=1)
    k_indices=np.argsort(distances)[:k]
    
    k_labels=y[k_indices]
    steps.append((k_indices, k_labels, distances))
    prediction = np.bincount(k_labels).argmax()
    return steps, prediction