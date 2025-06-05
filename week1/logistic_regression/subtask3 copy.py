import numpy as np
np.set_printoptions(precision=2,floatmode='unique')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# #CONVENTION FOR AXES -> (#dimensions of space(output or input), #datapoints)
# y:- (c,m) or (#classes , #datapoints)  NOTE:- for yes/no means that classes are 0 and 1. Expected NOTA option as the last class
# b:- (c,1)
# w:- (c,n) cuz w shud be a transformation matrix from n dimnesions to output c dimensions
def logistic_regression(X,Y, w=0,b_initial=0,learning_rate=9e0,max_iters=100, lambda_ = 0):
    (n,m) = X.shape
    if Y.shape[1]== X.shape[1]:
        (c,m) = Y.shape
    else:
        raise ValueError(f"Expected X.shape=(#features,#datapoints) and Y.shape=(#classes,#datapoints), got {X.shape} and{Y.shape}")
    if w==0:
        w=np.random.rand(c,n,)*10
    if b_initial==0:
        b=np.random.rand(c,1)*0
    if lambda_==0:
        lambda_ = learning_rate/20
    
    for iter in range(max_iters):
        #forward prop
        Z= w@X+b #(c.m)
        sum_cols = np.exp(Z).sum(axis=0,keepdims=1)#1,m
        Probabilities = np.exp(Z)/ sum_cols #(c,m)

        #backprop
        dLoss_dw = np.zeros(w.shape)
        dLoss_db = np.zeros(b.shape)
        for i in range(m):
            x = X[:,i].reshape(-1,1)
            y = Y[:,i].reshape(-1,1)
            #z = Z[:,i].reshape(-1,1) #(c,1)
            probabilities = Probabilities[:,i].reshape(-1,1)
            #sum_cols_i = sum_cols[0][i]

            dLoss_dw +=  ((probabilities-y) @ x.T)   # (c,1)  gives the d(loss of ith training example)/dz
            dLoss_db +=  ((probabilities-y) ) # (c,1)  gives the d(loss of ith training example)/dz

        dCost_dw = dLoss_dw/m
        #add l2 regularization cost
        dCost_dw += (2 * lambda_ / m) * w
        dCost_db = dLoss_db/m
        #grad step
        w -= learning_rate*dCost_dw
        b -= learning_rate*dCost_db
 

        if iter%int(max_iters/10)==0 or 1:
            print(f"dw = {-learning_rate*dCost_dw[0]}, db = {-learning_rate*dCost_db[0]}")
            print(f'w[0] = {w[0]}, b[0] = {b[0]}')

            Loss = -np.log(Probabilities+1e-100) * Y    #(c,m)
            cost = 1/m * Loss.sum() 
            regul_cost = lambda_/m*(w**2).sum()
            print(f"{cost+regul_cost:} = {cost:} + {regul_cost:}")

            argmaxed_predicted = np.argmax(Probabilities,axis=0)
            print(argmaxed_predicted)
            # predicted = np.zeros_like(Probabilities)
            # predicted[argmaxed, np.arange(Probabilities.shape[1])] = 1
            # print(predicted)
            argmaxed_actual = np.argmax(train_output,axis = 0)
            print(argmaxed_actual)


            percent_correct = np.mean(argmaxed_predicted == argmaxed_actual)
            print(f'Percentage classified correctly (train) :- {percent_correct*100}')
            print()
    return w,b


#columns are x,y,classification class (0 or 1)
train_data = np.genfromtxt("train3.csv", delimiter=',', names=True, dtype=None, encoding=None)
train_input = np.column_stack((train_data['x'],train_data['y'])).T.reshape(2,-1) #x,y are the rows
train_output = np.column_stack((train_data['color'],train_data['marker']))

#PRE PROCESS DATA
#ONE HOT ENCODE TRAIN OUTPUT
#train output is now a list of tuples [ ('red','^'), ...]
train_output = [tuple(row) for row in train_output]
#make a bijection bw classes (red and ^, blue and *) with an index
unique = list(set(train_output)) #list of unique tuples/classes
#bijection/dict bw a unique tuple and its one hot encoded vector
one_hot_encoding = {}
for i,class_  in enumerate(unique):
    #make array with 1 only at posiition i
    arr = np.zeros((len(unique)))
    arr[i] = 1
    one_hot_encoding[class_] = arr

train_output = np.array([one_hot_encoding[class_] for class_ in train_output]).T
#CENTRE INPUT
def make_quad(X):
    #X = np.vstack((X[0], X[1], X[0]**2, X[0]*X[1], X[1]**2, X[0]**3, X[0]**2*X[1],X[1]**2*X[0], X[1]**3))
    X = np.vstack((X[0], X[1], X[0]**2, X[0]*X[1], X[1]**2))
    return X
train_input = make_quad(train_input)

train_mean =train_input.mean(axis=1,keepdims=1)
train_std_dev = train_input.std(axis=1, keepdims=True)
def center_data(X):
    mean =train_mean
    std_dev = train_std_dev
    ans = (X-mean)/std_dev
    return ans

# X = train_input
# plt.plot(X[0],X[1],'bo')
# plt.show()
train_input= center_data(train_input)
X=train_input
# plt.plot(X[0],X[1],'bo')
# plt.show()

w,b =logistic_regression(train_input,train_output) #returns (c,n), (c,1)

#TEST
test_data = np.genfromtxt("test3.csv", delimiter=',', names=True, dtype=None, encoding=None)
test_input = np.column_stack((test_data['x'],test_data['y'])).T.reshape(2,-1) #x,y are the rows
test_output = np.column_stack((test_data['color'],test_data['marker']))

#PRE PROCESS DATA
#ONE HOT ENCODE TRAIN OUTPUT
#test output is now a list of tuples [ ('red','^'), ...]
test_output = [tuple(row) for row in test_output]

test_output = np.array([one_hot_encoding[class_] for class_ in test_output]).T
#CENTRE INPUT
# X=test_input
# plt.plot(X[0],X[1],'bo')
# plt.show()
test_input_orig = test_input
test_input = make_quad(test_input)
test_input= center_data(test_input)
# X=test_input
# plt.plot(X[0],X[1],'bo')
# plt.show()

Z = w@test_input +b
test_probabilities = np.exp(Z)/ np.exp(Z).sum(axis=0,keepdims=1)

argmaxed_predicted = np.argmax(test_probabilities,axis=0)
# print(argmaxed_predicted)
argmaxed_actual = np.argmax(test_output,axis = 0)
# print(argmaxed_actual)
percent_correct = np.mean(argmaxed_predicted == argmaxed_actual)
print(f'Percentage classified correctly (test):- {percent_correct*100}')

# percent_correct_dict = {}
# for i,color_marker in enumerate(unique):
#     prediction_i = prediction==unique[i]
#     test_output_i = test_output==unique[i]
#     percent_correct_i = np.sum(prediction_i==test_output_i)/ np.sum(test_output_i)
#     percent_correct_dict[unique[i]] = f"{percent_correct_i:.2f}"
# print(f'Percentage classified correctly per color:- {percent_correct_dict:.2%}')


#VISUALISER
#Plot actual points
test_output_argmax = np.argmax(test_output,axis=0)
color_marker = np.array([unique[idx] for idx in test_output_argmax])
colors =color_marker[:,0]
markers = color_marker[:,1]

for marker in np.unique(markers):
    idx = (markers == marker)
    plt.scatter(test_input_orig[0][idx],test_input_orig[1][idx], c=colors[idx], marker=marker, label=f"Marker {marker}")

#decision boundaries of each color
X= test_input_orig

buffer = 10
x_min, x_max = X[0].min()-buffer, X[0].max()+buffer
y_min, y_max = X[1].min()-buffer, X[1].max()+buffer
for i,color_marker in enumerate(unique):
    color_marker=np.array(color_marker)
    color =color_marker[0]
    marker = color_marker[1]
    #get y points o decision boundary, ST z = w_list[i].T@test_input +b_list[i] = 0
    num_points_in_line = 200
    X = np.linspace(x_min,x_max,num_points_in_line)
    X_transformed = (X-train_mean[0])/train_std_dev[0]
    Y_transformed = -(w[i][0]*X_transformed +b[i]) / w[i][1]
    Y = Y_transformed*train_std_dev[1] + train_mean[1]
    Y=Y.ravel()
    plt.plot(X, Y, color=color, linestyle='--', linewidth=1)

    #show weight arrow direcn and amount
    (dx,dy) = ((w[i][0]*train_std_dev[0]).item(),(w[i][1]*train_std_dev[1]).item())
    pointing = np.array( [dx,dy])
    pointing /= 1+0*np.linalg.norm(pointing)  # normalize
    dx, dy = pointing[0],pointing[1]  # ensure scalar
    idx_arrow = int(num_points_in_line/2)
    plt.arrow(X[idx_arrow], Y[idx_arrow], dx, dy, head_width=0.2, head_length=0.3, fc=color, ec=color, width=0.02)
plt.gca().set_aspect('equal') 


#heatmap of probability over input space
xx,yy = np.meshgrid(np.linspace(x_min,x_max,1000), np.linspace(y_min,y_max,1000))
grid_predict = np.vstack((xx.ravel(),yy.ravel()))
grid_predict = make_quad(grid_predict)
#predict grid output

grid_predict= center_data(grid_predict)

Z = w@grid_predict +b
Z = Z-Z.max(axis=0)
probabilities = np.exp(Z)/ np.exp(Z).sum(axis=0,keepdims=1) 
argmaxed_predicted = np.argmax(probabilities,axis=0)

class_colors = [color_marker[0] for color_marker in unique]  # from (color, marker)
cmap = ListedColormap(class_colors)

argmaxed_predicted=argmaxed_predicted.reshape(xx.shape)

plt.contourf(xx, yy, argmaxed_predicted, levels=np.arange(len(unique) + 1) - 0.5, cmap=cmap, alpha=0.3)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
















