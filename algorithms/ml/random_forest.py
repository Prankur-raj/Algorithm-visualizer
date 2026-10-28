from sklearn.ensemble import RandomForestClassifier
import numpy as np


def random_forest_steps(X, y, n_estimators=10, max_depth=3):
    steps = []

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X, y)

    predictions = model.predict(X)
    importances = model.feature_importances_

    # Capture OOB-style accuracy per tree count for animation
    for i in range(1, n_estimators + 1):
        partial_model = RandomForestClassifier(n_estimators=i, max_depth=max_depth)
        partial_model.fit(X, y)
        partial_preds = partial_model.predict(X)
        accuracy = np.mean(partial_preds == y)
        steps.append((i, accuracy, importances))

    return steps, model