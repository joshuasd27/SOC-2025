import numpy as np
np.set_printoptions(precision=2,floatmode='unique')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

np.random.seed(42)

"""
Sigmoid activation applied at each node.
"""
def sigmoid(x):
    # cap the data to avoid overflow?
    x[x>100] = 100
    x[x<-100] = -100
    return 1/(1+np.exp(-x))

"""
Derivative of sigmoid activation applied at each node.
"""
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class NN:
    def __init__(self, input_dim, hidden_dim, activation_func = sigmoid, activation_derivative = sigmoid_derivative):
        """
        Parameters
        ----------
        input_dim : (int) number of features in a single input
        hidden_dim : (int) number of nodes in hidden layer
        activation_func : function, optional
            Any function that is to be used as activation function. The default is sigmoid.
        activation_derivative : function, optional
            The function to compute derivative of the activation function. The default is sigmoid_derivative.

        Returns
        -------
        None.

        """
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        # Initialize weights and biases for the hidden and output layers
        self.w_1 = np.random.normal(loc=0,scale=1,size=(hidden_dim,input_dim))
        self.b_1 = np.zeros((hidden_dim,1))
        self.w_2 = np.random.normal(loc=0,scale=1,size=(1,hidden_dim))
        self.b_2 = np.zeros((1,1))
        # Init z's, activations for caching and usage in backprop
        self.z_1 = np.zeros((hidden_dim,input_dim))
        self.a_1 = np.zeros((hidden_dim,input_dim))
        self.z_2 = np.zeros((1,input_dim))
        self.a_2 = np.zeros((1,input_dim))
        
    def forward(self, X):
        """
        X:-(n,m) or (input_dim, no. of samples)
        """
        (n,m) =X.shape
        # Forward pass

        
        # Compute activations for all the nodes with the activation function applied 
        # for the hidden nodes, and the sigmoid function applied for the output node
        self.z_1 = self.w_1 @ X + self.b_1
        self.a_1 = sigmoid(self.z_1)        
        self.z_2 = self.w_2 @ self.a_1 + self.b_2
        self.a_2 = sigmoid(self.z_2)
        # Return: Output probabilities of shape (1, m) where m is number of examples
        return self.a_2
    
    def backward(self, X, y, learning_rate):
        """
        X:-(input_dim, no. of samples)
        y:-(1, no. of samples)
        """
        (n,m) = X.shape
        # Backpropagation
        # TODO: Compute gradients for the output layer after computing derivative of sigmoid-based binary cross-entropy loss
        # TODO: When computing the derivative of the cross-entropy loss, don't forget to divide the gradients by m (number of examples)  
        # TODO: Next, compute gradients for the hidden layer
        # TODO: Update weights and biases for the output layer with learning_rate applied
        # TODO: Update weights and biases for the hidden layer with learning_rate applied
        dCost_dz_2 = self.a_2 - y #(1,m)
        dCost_dw_2 = dCost_dz_2 @ self.a_1.T /m # (1,m) @ (m,hidden_dim) = (1,hidden_dim)
        dCost_db_2 = dCost_dz_2.mean(axis=1,keepdims=1) # (1,1)

        dCost_dz_1 = (dCost_dz_2 *(self.w_2.T *self.activation_derivative(self.z_1)) )#(1,m) * ((hidden_dim,1)*(hidden_dim,m))
        dCost_dw_1 = dCost_dz_1 @ X.T /m # (ihdden_dim,m) @ (m,n) = (hidden_dim,n)
        dCost_db_1 = dCost_dz_1.mean(axis=1,keepdims=1)  # (hidden_dims,1)
        
        self.w_2 -= learning_rate*dCost_dw_2
        self.b_2 -= learning_rate*dCost_db_2
        self.w_1 -= learning_rate*dCost_dw_1
        self.b_1 -= learning_rate*dCost_db_1

        
    
    def train(self, X, y, learning_rate, num_epochs):
        """
        X:-(input_dim, no. of samples)
        y:-(1, no. of samples)
        """
        for epoch in range(num_epochs):
            # Forward pass
            self.forward(X)
            # Backpropagation and gradient descent weight updates
            self.backward(X, y, learning_rate)
            # TODO: self.yhat should be an(1,m) vector containing the final
            # sigmoid output probabilities for all N training instances 
            self.yhat = self.a_2
            # TODO: Compute and print the loss (uncomment the line below)
            loss = np.mean(-y*np.log(self.yhat) - (1-y)*np.log(1-self.yhat))
            # TODO: Compute the training accuracy (uncomment the line below)
            accuracy = np.mean((self.yhat > 0.5).reshape(-1,) == y)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            self.pred('pred_train.txt')
            
    def pred(self,file_name='pred.txt'):
        pred = self.yhat > 0.5
        with open(file_name,'w') as f:
            for i in range(len(pred)):
                f.write(str(self.yhat[0][i]) + ' ' + str(int(pred[0][i])) + '\n')

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "data_train.csv"
    eval_file_path = "data_eval.csv"

    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)
    # Separate the data into X (features) and y (target) arrays
    X = data[:, :-1].T
    y = data[:, -1].reshape(1,-1)

    X_eval = data_eval[:, :-1].T
    y_eval = data_eval[:, -1].reshape(1,-1)

    # Create and train the neural network
    input_dim = X.shape[0]
    hidden_dim = 4
    learning_rate = 10
    num_epochs = 100
    
    model = NN(input_dim, hidden_dim)
    model.train(X**2, y, learning_rate, num_epochs) #trained on concentric circle data 

    test_preds = model.forward(X_eval**2)
    model.pred('pred_eval.txt')

    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Test accuracy: {test_accuracy:.4f}")
