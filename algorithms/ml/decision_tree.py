from sklearn.tree import DecisionTreeClassifier
import numpy as np
 
 
def decision_tree_steps(X, y, max_depth=3):
    steps = []
 
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X, y)
 
    predictions = model.predict(X)
    importances = model.feature_importances_
 
    steps.append((model, predictions, importances))
 
    return steps
 