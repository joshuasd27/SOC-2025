import numpy as np

#columns are x,y,classification class (0 or 1)
train_data = np.genfromtxt("train.csv", delimiter=',', skip_header=1)
train_input = train_data[:,0:2].T #x,y are the rows
train_output = train_data[:,2].reshape(1,-1)

test_data = np.genfromtxt("test.csv", delimiter=',', skip_header=1)
test_input = test_data[:,0:2].T #x,y are the rows
test_output = test_data[:,2].reshape(1,-1) # shape is 1,number of datapoints

# X:- (n,m) m datapoints, n features
# w:- (n,1) weights
#y:- (1,m) outputs
def logistic_regression(X,y,w_initial=np.array([[0],[0],[0.1],[0],[0]]),b_initial=0,learning_rate=0.0001, max_iters=1000):
    (n,m) = X.shape
    w=w_initial
    b=b_initial
    
    for iter in range(max_iters):
        z = w.T@X +b
        yPredicted= 1/(1+np.exp(-z))
        #calculate gradient
        dw =  ((2*(yPredicted - y) * X).sum(axis=1) / m ).reshape(-1,1)
        db = ((2*(yPredicted - y)).sum(axis=1) / m).reshape(-1,1)

        #grad step
        w -= learning_rate*dw
        b -= learning_rate*db


        if iter%int(max_iters/5)==0:
            cost =0
            for i in range(m):
                if y[0][i] ==1:
                    loss = -np.log(yPredicted[0][i])
                else:
                    loss= -np.log(1-yPredicted[0][i])
                cost+=loss
            cost /=m


            print(cost)
    print(w)
    return w,b

#boundary is quadratic ish
x_square = train_input[0]**2
xy = train_input[0]*train_input[1]
y_square = train_input[1]**2
quadratic_train_input =  np.vstack((train_input,x_square,xy,y_square))

w,b = logistic_regression(quadratic_train_input,train_output)

#TEST
x_square = test_input[0]**2
xy = test_input[0]*test_input[1]
y_square = test_input[1]**2
quadratic_test_input =  np.vstack((test_input,x_square,xy,y_square))
w,b = logistic_regression(quadratic_train_input,train_output)

#find cost
y=test_output
z = w.T@quadratic_test_input +b
yPredicted= 1/(1+np.exp(-z))
yPredicted= (yPredicted>0.5).astype(int)

print(f'Number of 1s:{y.sum()}')
print(f'Number of 0s:{y.shape[1]-y.sum()}')
print(f'Percentage of 1 classified correctly : {(y.astype(bool)&yPredicted.astype(bool)).sum()/y.sum():.2%}')
print(f'Percentage of 0 classified correctly : {((~y.astype(bool))&(~yPredicted.astype(bool))).sum()/(y.shape[1]-y.sum()):.2%}')
print(f'Total percentage classified correctly : {(y==yPredicted).sum()/y.shape[1]:.2%}')



