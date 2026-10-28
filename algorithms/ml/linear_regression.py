import numpy as np

def linear_regression_steps(x,y,epochs=50,lr=0.01):
    m=b=0
    steps=[]
    for i in range(epochs):
        y_pred=m*x+b

        dm=(-2/len(x)*np.sum(x*(y-y_pred)))
        db=(-2/len(x)*np.sum((y-y_pred)))

        m=m-lr*dm
        b=b-lr*db

        steps.append((m,b))

    return steps