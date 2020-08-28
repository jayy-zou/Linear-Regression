import numpy as np

def mean_squared_error(estimates, targets):
    sum=0
    for i in range(len(targets)):
        sum+=(estimates[i]-targets[i])**2

    return (1/np.shape(estimates)[0])*sum
