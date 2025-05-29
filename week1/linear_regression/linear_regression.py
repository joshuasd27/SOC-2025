import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("smol_dataset.csv", delimiter=",", skip_header=1, usecols=range(1, 6))
X= data[:,0:3]
X= X.T #X has (n.m) n features m datapoints
mangos = data[:,3]
mangos = mangos.reshape(1,-1)
oranges = data[:,4]
oranges = oranges.reshape(1,-1)


# X:- (n,m) m datapoints, n features
# w:- (n,1) weights
#y:- (1,m) outputs
def linear_regression(X,y,w_initial=0,b_initial=0,learning_rate=0.00003):
    (n,m) = X.shape
    if w_initial==0:
        w=np.ones((n,1))
    b=b_initial
    

    max_iters = 1000
    for iter in range(max_iters):
        yPredicted = w.T@X +b
        #calculate gradient
        dw =  ((2*(yPredicted - y) * X).sum(axis=1) / m ).reshape(-1,1)
        db = ((2*(yPredicted - y)).sum(axis=1) / m).reshape(-1,1)

        #grad step
        w -= learning_rate*dw
        b -= learning_rate*db

        # if iter%1==0:
        #     cost = ((yPredicted-y)**2).sum() / m
        #     if cost<1e10:
        #         print(cost)
    
    return w,b

w,b = linear_regression(X,mangos)
print(w)
print(b)
