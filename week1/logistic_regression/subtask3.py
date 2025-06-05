import numpy as np
np.set_printoptions(precision=2,floatmode='unique')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#columns are x,y,classification class (0 or 1)
train_data = np.genfromtxt("train3.csv", delimiter=',', names=True, dtype=None, encoding=None)
train_input = np.column_stack((train_data['x'],train_data['y'])).T.reshape(2,-1) #x,y are the rows
train_output = np.column_stack((train_data['color'],train_data['marker']))


#CONVENTION FOR AXES -> (#dimensions of space(output or input), #datapoints)

# y:-(c,m)  (#classes , #datapoints) NOTE:- for yes/no means that classes are 0 and 1. Expected NOTA option as the last class
# b:-(c,1)
# w:-(c,n) cuz w shud be a transformation matrix from n dimnesions to output c dimensions
def logistic_regression(X,Y, w=None,b_initial=None,learning_rate=0.001,max_iters=20, lambda_ = 0):
    (n,m) = X.shape
    if Y.shape[1]== X.shape[1]:
        (c,m) = Y.shape
    else:
        raise ValueError(f"Expected X.shape=(#features,#datapoints) and Y.shape=(#classes,#datapoints), got {X.shape} and{Y.shape}")
    if w==None:
        w=np.random.rand(c,n,)*0.01
    if b_initial==None:
        b=np.random.rand(c,1)*0
    if lambda_==0:
        lambda_ = learning_rate/10

    for iter in range(max_iters):
        Z= w@X+b
        Sigmoid = 1/(1+np.exp(-Z)) +1e-15
        sum_cols = Sigmoid.sum(axis=0,keepdims=1)
        Probabilities = Sigmoid/sum_cols
        Loss = -np.log(Probabilities+1e-15) * Y
        cost = 1/m * Loss.sum() 
        dLoss_dw = np.zeros(w.shape)
        dLoss_db = np.zeros(b.shape)
        for i in range(m):
            X_i = X[:,i].reshape(-1,1)
            y_i = Y[:,i].reshape(-1,1)
            z = Z[:,i].reshape(-1,1) #(c,1)
            a = Sigmoid[:,i].reshape(Sigmoid.shape[0],-1)     #(c,1)    
            probabilities = Probabilities[:,i].reshape(-1,1)
            sum_cols_i = sum_cols[0][i]

            #index of crct answer/one hot encoding
            idx = np.argmax(y_i)
            dLoss_dw +=  -1/probabilities[idx]*((sum_cols_i*y_i-a)/sum_cols_i**2).T @ (a*(1-a) @ X_i.T)  # (c,1)  gives the d(loss of ith training example)/dz
            dLoss_db +=  -1/probabilities[idx]*((sum_cols_i*y_i-a)/sum_cols_i**2).T @ (a*(1-a))  # (c,1)  gives the d(loss of ith training example)/dz

        dCost_dw = dLoss_dw/m
        dCost_dw += (2 * lambda_ / m) * w
        dCost_db = dLoss_db/m
        print(f"dw = {learning_rate*dCost_dw[0]}, db = {learning_rate*dCost_db[0]}")
        #grad step
        w -= learning_rate*dCost_dw
        b -= learning_rate*dCost_db
        print(w[0])
        print(b[0])

        if iter%int(max_iters/10)==0 or 1:
            regul_cost = lambda_/m*(w**2).sum()
            print(f"{cost+regul_cost:} = {cost:} + {regul_cost:}")
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
y = np.array([one_hot_encoding[class_] for class_ in train_output]).reshape(len(unique),-1)

mean =train_input.mean(axis=1,keepdims=1)
std_dev = ((train_input -mean)**2).mean(axis=1, keepdims=1)
train_input = (train_input-mean)/std_dev

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
mean =X.mean(axis=1,keepdims=1)
std_dev = ((X -mean)**2).mean(axis=1,keepdims=1)
X = (X-mean)/std_dev

y= test_output
z = w@X +b
y_probability_distri = 1/(1+np.exp(-z))
y_probability_distri /= y_probability_distri.sum(axis=0,keepdims=1)

prediction = np.array( [np.argmax(y_probability_distri[:,i]) for i in range(test_input.shape[1])] )
prediction = [tuple(unique[arg]) for i,arg in enumerate(prediction)]
#make into list of tuples

# print(prediction)
# print(test_output)
percent_correct = np.mean([x == y for x, y in zip(prediction, test_output)])
print(f'Percentage classified correctly :- {percent_correct:.2%}')

# percent_correct_dict = {}
# for i,color_marker in enumerate(unique):
#     prediction_i = prediction==unique[i]
#     test_output_i = test_output==unique[i]
#     percent_correct_i = np.sum(prediction_i==test_output_i)/ np.sum(test_output_i)
#     percent_correct_dict[unique[i]] = f"{percent_correct_i:.2f}"
# print(f'Percentage classified correctly per color:- {percent_correct_dict:.2%}')


#VISUALISER
#Plot actual points
color_marker = np.array(test_output)
colors =color_marker[:,0]
markers = color_marker[:,1]

for marker in np.unique(markers):
    idx = markers == marker
    plt.scatter(X[0][idx],X[1][idx], c=colors[idx], marker=marker, label=f"Marker {marker}")


#decision boundaries of each color
x_min, x_max = X[0].min()-1, X[0].max()+1
y_min, y_max = X[1].min()-1, X[1].max()+1
for i,color_marker in enumerate(unique):
    color_marker=np.array(color_marker)
    print(color_marker)
    color =color_marker[0]
    marker = color_marker[1]
    #get y points o decision boundary, ST z = w_list[i].T@test_input +b_list[i] = 0
    X = np.linspace(x_min,x_max,200)
    Y = -(w[i][0]*X +b[i]) / w[i][1]
    Y=Y.ravel()
    plt.scatter(X,Y, c=color, marker=marker)
plt.show()














