import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#columns are x,y,classification class (0 or 1)
train_data = np.genfromtxt("train3.csv", delimiter=',', names=True, dtype=None, encoding=None)
train_input = np.column_stack((train_data['x'],train_data['y'])).T.reshape(2,-1) #x,y are the rows
train_output = np.column_stack((train_data['color'],train_data['marker']))


# X:- (n,m) m datapoints, n features
# y:- (1,m) outputs
# w:- (n,1) initial weights
# b:- (1,1)

#make it handle not just binary but multi class classification
# y^ needs to be a multi class precdiction :- {red, green, blue} = {0.5,0.25.0.25}
#CONVENTION FOR AXES -> (#dimensions of space(output or input), #datapoints)
# 0.5 comes from w_red*X+b_red etc
# y:- (#classes , #datapoints) or (c,m) NOTE:- for yes/no means that classes are 0 and 1. Expected NOTA option as the last class
# b now is (c,1)
# w is now (c,n) cuz w shud be a transformation matrix from n dimnesions to output c dimensions
def logistic_regression(X,y, w=0,b_initial=0,learning_rate=0.035, max_iters=100):
    (n,m) = X.shape
    if y.shape[1]== X.shape[1]:
        (c,m) = y.shape
    else:
        raise ValueError(f"Expected X.shape=(#features,#datapoints) and Y.shape=(#classes,#datapoints), got {X.shape} and{Y.shape}")
    if w==0:
        w=np.zeros((c,n))
    if b_initial==0:
        b=np.zeros((c,1))
    
    for iter in range(max_iters):
        z = w@X +b
        yPredicted= 1/(1+np.exp(-z))
        #calculate gradient of cost/average suprise 
        # dw =  ((2*(yPredicted - y) * X).sum(axis=1) / m ).reshape(-1,1)
        # db = ((2*(yPredicted - y)).sum(axis=1) / m).reshape(-1,1)
        dw = np.zeros(w.shape)
        db=0

        Z= w@X+b
        Sigmoid = 1/(1+np.exp(-z))  
        for i in range(m):
            #X_i = X[:,i].reshape(X.shape[0],-1)
            y_i = y[:,i].reshape(y.shape[0],-1)
            #z = w@X_i +b 
            z = Z[:,i].reshape(Z.shape[0],-1) #(c,1)
            sigmoid = Sigmoid[:,i].reshape(Sigmoid.shape[0],-1)     #(c,1)    
            sum_sigmoid = sigmoid.sum()  
            #loss/suprise is sum(y[i] * ln(1/probability of class i)) for a training example
            #prob of class i is sigmoid / sigmoid.sum()

            dLoss_dz = (y_i *(1/sum_sigmoid-1/sigmoid)) * sigmoid*(1-sigmoid)  # (c,1)  gives the d(loss of ith training example)/dz
            dw+= dLoss_dz * X.T[i]   # (c,1) * (1,n) = (c,n) broadcasting gives the d(loss of ith training example)/dw in w.shape form
            db += dLoss_dz
        dw/=m
        db/=m

        #grad step
        w -= learning_rate*dw
        b -= learning_rate*db


        if iter%int(max_iters/5)==0:
            cost =0
            Z= w@X+b
            Sigmoid = 1/(1+np.exp(-Z))  

            for i in range(m):
                y_i = y[:,i].reshape(y.shape[0],-1)
                #z = w@X_i +b 
                z = Z[:,i].reshape(Z.shape[0],-1) #
                sigmoid = Sigmoid[:,i].reshape(Sigmoid.shape[0],-1)
                sum_sigmoid = sigmoid.sum()  
                cost += np.mean(y_i * (-np.log(sigmoid/sum_sigmoid)))
            cost /= m


            print(cost)
    #print(w)
    return w,b

#PRE PROCESS DATA
#make classes as numpy is a fucking asshole about treating a 1d arry as an object -> use tuple
#train_output now a list 
train_output = [tuple(row) for row in train_output]
#map class to a number
unique = list(set(train_output))

one_hot_encoding = {}
for i,class_  in enumerate(unique):
    #make array with 1 only at posiition i
    arr = np.zeros(len(unique))
    arr[i] = 1
    one_hot_encoding[class_] = arr

#print(train_output)
y = np.array([one_hot_encoding[class_] for class_ in train_output]).T

w,b =logistic_regression(train_input,y) #returns (c,n), (c,1)



#TEST
test_data = np.genfromtxt("test3.csv", delimiter=',', names=True, dtype=None, encoding=None)
test_input = np.column_stack((test_data['x'],test_data['y'])).T.reshape(2,-1) #x,y are the rows
test_output = np.column_stack((test_data['color'],test_data['marker']))
#pre process
#train_output now a list encoded with index of unique that it is
test_output = [tuple(row) for row in test_output]

#predict
X=test_input
y= test_output
z = w@X +b
y_probability_distri = 1/(1+np.exp(-z))
y_probability_distri /= y_probability_distri.sum(axis=0)

prediction = np.array([np.argmax(y_probability_distri[:,i]) for i in range(test_input.shape[1])])
prediction = [tuple(unique[arg]) for i,arg in enumerate(prediction)]
#make into list of tuples

percent_correct = np.mean([x == y for x, y in zip(prediction, test_output)])
print(f'Percentage classified correctly :- {percent_correct:.2%}')


# #VISUALISER
# #Plot actual points
# plt.scatter(test_input[0],test_input[1], c=test_output, cmap='bwr', edgecolors='b', s=50)


# #decision boundaries of each color
# x_min, x_max = test_input[0].min()-1, test_input[0].max()+1
# y_min, y_max = test_input[1].min()-1, test_input[1].max()+1
# for i,color in enumerate(unique):
#     #get y points o decision boundary, ST z = w_list[i].T@test_input +b_list[i] = 0
#     X = np.linspace(x_min,x_max,200)
#     Y = -(w_list[i][0]*X +b_list[i]) / w_list[i][1]
#     Y=Y.ravel()
#     plt.plot(X,Y, c=color)
# plt.show()














