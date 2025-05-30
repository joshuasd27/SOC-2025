import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#columns are x,y,classification class (0 or 1)
train_data = np.genfromtxt("train2.csv", delimiter=',', names=True, dtype=None, encoding=None)
train_input = np.column_stack((train_data['x'],train_data['y'])).T #x,y are the rows
train_output = train_data['color']
#map color to a number
unique, counts = np.unique(train_output, return_counts=1)
frequency = dict(zip(unique,counts))
color_dict = {color:i for i,color in enumerate(unique)}
#change colors to coreresp numbes
map_func = np.vectorize(color_dict.get)
#train_output_num = map_func(train_output).reshape(1,-1)


def logistic_regression(X,y,w=0,b_initial=0,learning_rate=0.5, max_iters=3000):
    (n,m) = X.shape
    if w==0:
        w=np.zeros((n,1))
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


w_list =[]
b_list = []
for i,color in enumerate(unique):
    y= (train_output==color).astype(float).reshape(1,-1)
    w_colori,b_colori =logistic_regression(train_input,y)
    print(train_input.shape)
    print(y.shape)
    w_colori= w_colori.reshape(-1,1)
    w_list.append(w_colori)
    b_list.append(b_colori)


#TEST
test_data = np.genfromtxt("test2.csv", delimiter=',', names=True, dtype=None, encoding=None)
test_input = np.column_stack((test_data['x'],test_data['y'])).T #x,y are the rows
test_output = test_data['color']
#predict
yPredicted_list= []
for i,color in enumerate(unique):
    y= (test_output==color).astype(float).reshape(1,-1)
    z = w_list[i].T@test_input +b_list[i]
    yPredicted= 1/(1+np.exp(-z))
    yPredicted_list.append(yPredicted)
yPredicted_arr = np.array(yPredicted_list)
prediction = np.array([np.argmax(yPredicted_arr[:,0,i]) for i in range(test_input.shape[1])])
prediction = np.array( [unique[arg] for i,arg in enumerate(prediction)])


percent_correct = np.mean(test_output==prediction)

print(f'Distribution of data points')
print(frequency)
print(f'Percentage classified correctly :- {percent_correct:.2%}')


#VISUALISER
#Plot actual points
plt.scatter(test_input[0],test_input[1], c=test_output, cmap='bwr', edgecolors='b', s=50)


#decision boundaries of each color
x_min, x_max = test_input[0].min()-1, test_input[0].max()+1
y_min, y_max = test_input[1].min()-1, test_input[1].max()+1
for i,color in enumerate(unique):
    #get y points o decision boundary, ST z = w_list[i].T@test_input +b_list[i] = 0
    X = np.linspace(x_min,x_max,200)
    Y = -(w_list[i][0]*X +b_list[i]) / w_list[i][1]
    Y=Y.ravel()
    plt.plot(X,Y, c=color)
plt.show()














