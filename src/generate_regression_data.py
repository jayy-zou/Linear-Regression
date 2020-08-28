import numpy as np


def generate_regression_data(degree, N, amount_of_noise=1.0):
    x=np.random.uniform(-1,1,N)

    coeffs=np.random.uniform(-10.0,10.0,degree+1)

    y=np.zeros(N)

    for i in range(N):
        sum=0
        for coeff in range(degree+1):
            sum+=coeffs[coeff]*(x[i]**coeff)
        y[i]=sum

    return x, (y+np.random.normal(0.0,amount_of_noise,y.shape))
