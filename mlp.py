import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
'''Helper Methods.'''
def random_initalizer(n, m):
    """
    :param: n height
    :param: m width 
    :return: (nxm) random matrix
    """
    return 0.01*np.random.randn(n, m)

def random_optimized_initalizer(n, m):
    """
    :param: n height
    :param: m width 
    :return: (nxm) random matrix
    """
    return 0.01*np.random.randn(n, m) * np.sqrt(2.0 / n) #Recommended here: https://cs231n.github.io/neural-networks-2/#datapre

def ReLU(z):
    """ReLU activation function."""
    return np.maximum(0, z)

def ReLU_derivative(a):
    """Derivative of ReLU activation."""
    a[a<= 0] = 0
    a[a > 0] = 1
    return a #Returns 1 if true otherwise 0 recommended the epsilon for numerical stability

def sigmoid(z):
    """Sigmoid activation function."""
    return 1./(1 + np.exp(-z))

def softmax_derivative(a):
    """Derivative of Sigmoid activation."""
    return sigmoid(a) * (1 - sigmoid(a))

def softmax(z):
    """Softmax activation for the output layer."""
    """NOTE: textbook pg. 438"""

    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  #Log-sum-exp trick
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def softmax_derivative(a):
    """
    Derivative for softmax activation function.
    """
    return softmax(a)*softmax(1 - a)

class ActivationFunction:
    """
    Dynamic class to handle activation functions and their derivatives
    """
    def __init__(self, func, derivative):
            self.func = func
            self.derivative = derivative

    def __call__(self, x):
        return self.func(x)

    def grad(self, x):
        return self.derivative(x)

class MLPSoftmax:
    """
    MLP Softmax class
    """
    def __init__(self, initalizer, loss_activation,  activation_function, layer_sizes, normalization = True):
        """
        Initialize the MLP with random weights and biases.
        :param layer_sizes: List of layer sizes. [input_dim, hidden_1, ..., hidden_k, output_dim] Example: [D, 64, 64, 11]
        """
        self.initalizer = initalizer
        self.loss_activation = loss_activation
        self.activation_function = activation_function
        self.normalization = normalization
        self.K = len(layer_sizes) - 1  # Number of layers excluding input
        self.weights = [self.initalizer(layer_sizes[i], layer_sizes[i + 1]) for i in range(self.K)] # Ex. w_0 (D, Unit_1), w_1 (Unit_1, Unit_2), w_2 (Unit_2, C)
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(self.K)] #https://cs231n.github.io/neural-networks-2/#datapre Recommends initalize bias as zero

    
    
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        """Compute the cross-entropy loss. Assume y_true is OHE."""
        """NOTE: We choose cross-entropy loss to simpify the Jacobian as described in slide chapter 9, 12.
        and textbook pg. 438"""
        #m = y_true.shape[0]
        N = y_pred.shape[0]
        y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / N #Stability reasons recommended by ChatGPT

    def normalize(self, X):
        """
        Normalize the data. 
        :param X: Input data assuming (N x D) D is the vectors dimensionality
        From: https://cs231n.github.io/neural-networks-2/#datapre
        """
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis= 0)
        return X

    def batch_normalization(self, a, gamma=1.0, beta =0.0, eps=1e-5):
        """
        Batch normalization to help control high weights/reguralize.
        :param a: The input activation layer
        :param gamma: hyper-parameter to control normalization (Assume default) 
        :param beta: hyper-paramaeter to control normalization (Assume default)
        """
        mean = np.mean(a, axis=0, keepdims=True)
        variance = np.var(a, axis=0, keepdims=True)
        x_norm = (a - mean) / np.sqrt(variance + eps)
        return gamma * x_norm + beta
    
    def clip_gradients(grads, clip_value):
        """
        Implement gradient clipping 
        """
        return 0
    
    
    def forward(self, X):
        """
        Perform forward propagation.
        :param X: Input data of shape (n_samples, input_dim).
        :return: Activations and linear combinations for each layer.
        """
        activations = [X] #Begin at input X (N x D)
        logits = []
        # i = 0
        for w, b in zip(self.weights[:-1], self.biases[:-1]): #Since we don't want to include the cross-entropy layer
            z = activations[-1] @ w + b #w_0 (D x Unit_1), #w_1 (Unit_1 x Unit_2), ... Thus z is always (N, Unit_i) size
            logits.append(z) # (N, Unit_i)
            a = self.activation_function(z) 
            a = self.batch_normalization(a)
            activations.append(a) # every (N, Unit_i)
            # i += 1
            # print(f"Layer {i}, Max Activation: {np.max(a)}, Min Activation: {np.min(a)}")

        
        # Output layer
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        logits.append(z) #(N, Unit_i) (Unit_i, C)
        a = self.loss_activation(z) #(N, C)
        activations.append(a)
        return activations, logits
    
    def backward(self, X, y, activations, logits):
        """
        Perform backward propagation.
        :param X: Input data.
        :param y: True labels (one-hot encoded).
        :param activations: Activations from the forward pass.
        :param Logits: Linear combinations from the forward pass.
        :return: Gradients for weights and biases.
        """
        N = X.shape[0]
        error_above = (activations[-1] - y)*self.loss_activation(activations[-1])  # Output layer error (Previous layer error) our K. Last being index -1 DIM(N, Unit_i)
        weight_grads = []
        bias_grads = []
        
        for i in range(self.K-1, -1, -1): # i = K - 1, K - 2, K - 3, ...
            dW = activations[i].T @ error_above / N # (N, Unit_i)T @ (N, Unit_i) => (Unit_i, N) @ (N, Unit_i)
            db = np.sum(error_above, axis=0, keepdims=True) / N #May remove this in favor of biases in weights matrices
            weight_grads.insert(0, dW)
            bias_grads.insert(0, db)
            # print(f"Layer {i}, Max Weight error: {np.max(error_above)}, Min Weight error: {np.min(error_above)}")
            # print(f"Layer {i}, Max Weight Grad: {np.max(dW)}, Min Weight Grad: {np.min(dW)}")

            
            if i > 0:  # Compute error for the previous layer
                """NOTE: The Hadamard product here we use is equivalent to the diagonal derivative matrix described in the textbook."""
                error_above = (error_above @ self.weights[i].T) * self.activation_function.grad(logits[i - 1]) # ((N, Unit_i) (Unit_{i - 1}, Unit_i)T) * (N, Unit_{i - 1}) NOTE: i - 1 because logits contains the output logit
        
        return weight_grads, bias_grads
    
    def update_parameters(self, weight_grads, bias_grads, learning_rate, t, scheduler_p=0.5):
        """
        Update weights and biases using gradients.
        :param weight_grads: Gradients for weights.
        :param bias_grads: Gradients for biases.
        :param learning_rate: Initial learning rate for parameter updates.
        :param iterations: Current iteration for the scheduler. 
        :param scheduler_p: Parameter for the scheduler
        """
        #learning_rate = learning_rate / (1 + t ** scheduler_p)
        for i in range(self.K):

            assert(len(self.weights) == len(weight_grads))
            assert(len(self.weights[i]) == len(weight_grads[i]))
            assert(len(self.biases) == len(bias_grads))
            assert(len(self.biases[i]) == len(bias_grads[i]))

            self.weights[i] = self.weights[i] - learning_rate * weight_grads[i]
            #print(f"difference of weights to grads: {np.linalg.norm(self.weights[i]) - np.linalg.norm(weight_grads[i])}")
            self.biases[i] = self.biases[i] - learning_rate * bias_grads[i]
    
    def fit(self, X_train, y_train, X_test, y_test, learning_rate, epochs, batch,  termination_condition, max_iters, plot = False):
        """
        Train the MLP using gradient descent.
        :param X: Input data.
        :param y: True labels (one-hot encoded).
        :param learning_rate: Learning rate.
        :param epochs: Number of training iterations.
        """
        N, D = X_train.shape
        grad_norm = np.inf
        x_log = []
        test_log = []
        los = []
        for epoch in range(epochs):
            iterations = 0
            seed = np.arange(X_train.shape[0])
            np.random.shuffle(seed)
            x_ = X_train[seed]
            y_ = y_train[seed]
            for i in range(int(N / batch)):
                k = i * batch
                j = (i + 1)*batch
                activations, logits = self.forward(x_[k:j])
                weight_grads, bias_grads = self.backward(x_[k:j], y_[k:j], activations, logits)
                self.update_parameters(weight_grads, bias_grads, learning_rate, t=iterations)

                grad_norm = np.linalg.norm(np.hstack([g.ravel() for g in weight_grads + bias_grads])) #Compute norm over all our gradients
                if iterations >= max_iters: #Conditional Check for terminations 
                    break
                iterations += 1

        
            x_log.append(self.evaluate_acc(np.argmax(y_train, axis= 1), self.predict(X_train)))
            test_log.append(self.evaluate_acc(np.argmax(y_test, axis=1), self.predict(X_test)))
        #Plot within the function
        if plot:
            print("Epoch", epoch)
            print("Train accuracy:", x_log[-1])
            print("Test accuracy:", test_log[-1])
            plt.plot(x_log, label = 'train accuracy')
            plt.plot(test_log, label='test accuracy')
            plt.legend(loc = 'best')
            plt.ylabel('Accuracy')
            plt.xlabel('epoches')
            plt.grid()
            plt.show()

    
    def predict(self, X):
        """
        Perform forward propagation.
        :param X: Input data of shape (n_samples, input_dim).
        :param y: True labels. (n_samples, Classes)
        :return: loss / or actual label
        """
        activations, _ = self.forward(X)
        output_probabilites = activations[-1]
        return np.argmax(output_probabilites, axis=1) #Assuming that the index corresponds to the class OHE


    def evaluate_acc (self, y_true, y_pred):
        """
        To interpret evaluation better.
        """
        return np.sum(y_true == y_pred)/y_pred.shape[0]



if __name__ == "__main__":
    # # Random seed for reproducibility
    # np.random.seed(42)
    
    # # # Generate dummy data
    # X = np.random.rand(1000, 10)  # 100 samples, 3 features
    # print(X.shape)
    # y_raw = np.random.randint(0, 2, 1000)  # Random integer labels for 3 classes
    # y = np.eye(2)[y_raw]  # Convert to one-hot encoding
    # print(y.shape)



    # Load the wine dataset
    data = load_wine()

    # Extract features (X) and target labels (y) as numpy arrays
    X = data.data # 173, 13
    y = data.target # 173, 3
    y = np.eye(3)[y]
    y_list = np.argmax(y, axis= 1)



    # Perform 80-20 split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    Y_train, y_test = train_test_split(y, test_size= 0.2, random_state=42)








    
    # Initialize and train the MLP
    loss_activation_function = ActivationFunction(func = softmax, derivative= softmax_derivative)
    activation_function_relu = ActivationFunction(func = ReLU, derivative= ReLU_derivative)
    mlp = MLPSoftmax(initalizer= random_optimized_initalizer,loss_activation= loss_activation_function,  activation_function= activation_function_relu, layer_sizes= [13, 256, 3])  # Input: 3, Hidden: 5, Output: 3 (softmax)
    print(f"cross-entropy y_label againt y_label: {mlp.cross_entropy_loss(y, y)}")
    print(f"Intialized weights loss: {mlp.predict(X)}")
    mlp.fit(X_train=X, y_train= y,X_test=X_test, y_test=y_test, learning_rate=1e-4, epochs=50, batch= 10, termination_condition= 1e-3, max_iters= 1000, plot=True)
    predictions = mlp.predict(X)
    print(f"Final Loss: {predictions}")
    print(f"True labels: {y_list}")
    print(f"Final Accuracy:{mlp.evaluate_acc(y_true = y_list, y_pred= predictions)}")
