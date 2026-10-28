from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
import numpy as np


def logistic_regression_steps(X, y, max_iter=100):
    steps = []

    # Train incrementally to capture convergence
    for i in range(1, max_iter + 1, 10):
        model = LogisticRegression(max_iter=i, solver='lbfgs',)
        try:
            model.fit(X, y)
            predictions = model.predict(X)
            accuracy = np.mean(predictions == y)
            probabilities = model.predict_proba(X)
            steps.append((predictions, accuracy, probabilities, model))
        except Exception:
            continue

    # Make sure we always have at least one step
    if len(steps) == 0:
        model = LogisticRegression(max_iter=200, solver='lbfgs')
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        probabilities = model.predict_proba(X)
        steps.append((predictions, accuracy, probabilities, model))

    return steps