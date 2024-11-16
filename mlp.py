import numpy as np
from sklearn.datasets import load_wine
import math

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
    return 0.01*np.random.randn(n, m) * math.sqrt(2.0 / n) #Recommended here: https://cs231n.github.io/neural-networks-2/#datapre

def ReLU(z):
    """ReLU activation function."""
    return np.maximum(0, z)

def ReLU_derivative(a):
    """Derivative of ReLU activation."""
    return (a > 0).astype(np.float32) + 1e-10 #Returns 1 if true otherwise 0 recommended the epsilon for numerical stability

def sigmoid(z):
    """Sigmoid activation function."""
    return 1./(1 + np.exp(-z))

def softmax_derivative(a):
    """Derivative of Sigmoid activation."""
    return sigmoid(a) * (1 - sigmoid(a))

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
    def __init__(self, initalizer, activation_function, layer_sizes, normalization = True):
        """
        Initialize the MLP with random weights and biases.
        :param layer_sizes: List of layer sizes. [input_dim, hidden_1, ..., hidden_k, output_dim] Example: [D, 64, 64, 11]
        """
        self.initalizer = initalizer
        self.activation_function = activation_function
        self.normalization = normalization
        self.K = len(layer_sizes) - 1  # Number of layers excluding input
        self.weights = [self.initalizer(layer_sizes[i], layer_sizes[i + 1]) for i in range(self.K)] # Ex. w_0 (D, Unit_1), w_1 (Unit_1, Unit_2), w_2 (Unit_2, C)
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(self.K)] #https://cs231n.github.io/neural-networks-2/#datapre Recommends initalize bias as zero
        #Concatenate these two 

    
    @staticmethod
    def softmax(z):
        """Softmax activation for the output layer."""
        """NOTE: I will assume that we have the nice cross-entropy layer as described in textbook pg. 438, 
        thus not needing to calculate the diagonal jacobian of softmax."""
        # Clip z to prevent overflow

        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  #Log-sum-exp trick
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        """Compute the cross-entropy loss. Assume y_true is OHE."""
        """NOTE: We choose cross-entropy loss to simpify the Jacobian as described in slide chapter 9, 12.
        and textbook pg. 438"""
        #m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8))#Stability reasons recommended by ChatGPT

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
        a = self.softmax(z) #(N, C)
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
        error_above = activations[-1] - y  # Output layer error (Previous layer error) our K. Last being index -1 DIM(N, Unit_i)
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
                l = logits[i - 1]
                diag = self.activation_function.grad(logits[i - 1])
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
        learning_rate = learning_rate / (1 + t ** scheduler_p)
        for i in range(self.K):

            assert(len(self.weights) == len(weight_grads))
            assert(len(self.weights[i]) == len(weight_grads[i]))
            assert(len(self.biases) == len(bias_grads))
            assert(len(self.biases[i]) == len(bias_grads[i]))

            self.weights[i] = self.weights[i] - learning_rate * weight_grads[i]
            #print(f"difference of weights to grads: {np.linalg.norm(self.weights[i]) - np.linalg.norm(weight_grads[i])}")
            self.biases[i] = self.biases[i] - learning_rate * bias_grads[i]
    
    def fit(self, X, y, learning_rate, epochs, batch,  termination_condition, max_iters):
        """
        Train the MLP using gradient descent.
        :param X: Input data.
        :param y: True labels (one-hot encoded).
        :param learning_rate: Learning rate.
        :param epochs: Number of training iterations.
        """
        N, D = X.shape
        # if self.normalization:
        #     X = self.normalize(X) 
        grad_norm = np.inf
        #loss = 0
        for epoch in range(epochs):
            iterations = 0
            for i in range(int(N / batch)):
                random_indices = np.random.choice(N, batch, replace=False)
                X_batch = X[random_indices, :]
                y_batch = y[random_indices, :]
                activations, logits = self.forward(X_batch)
                weight_grads, bias_grads = self.backward(X_batch, y_batch, activations, logits)

                grad_norm = np.linalg.norm(np.hstack([g.ravel() for g in weight_grads + bias_grads])) #Compute norm over all our gradients
                if grad_norm <= termination_condition or iterations >= max_iters: #Conditional Check for terminations 
                    print(f"Stopping early at iteration {iterations + 1}")
                    break
                self.update_parameters(weight_grads, bias_grads, learning_rate, t=iterations)
                iterations += 1
                #loss = self.cross_entropy_loss(y_batch, activations[-1])
                #print(f"Batch{i + 1}, Loss: {loss:.4f}")
                #print(f"Batch: {i} Max value in weights:", np.max(self.weights[0]))

                if grad_norm <= termination_condition or iterations >= max_iters:
                    break
                #print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    
    def predict(self, X, y):
        """
        Perform forward propagation.
        :param X: Input data of shape (n_samples, input_dim).
        :param y: True labels. (n_samples, Classes)
        :return: loss
        """
        N = X.shape[0]
        activations, _ = self.forward(X)
        output_probabilites = activations[-1]
        loss = self.cross_entropy_loss(y, output_probabilites) / N
        return loss, output_probabilites



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







    
    # Initialize and train the MLP
    activation_function_relu = ActivationFunction(func = ReLU, derivative= ReLU_derivative)
    mlp = MLPSoftmax(initalizer= random_optimized_initalizer, activation_function= activation_function_relu, layer_sizes= [13, 256, 3])  # Input: 3, Hidden: 5, Output: 3 (softmax)
    print(f"cross-entropy y_label againt y_label: {mlp.cross_entropy_loss(y, y)}")
    print(f"Intialized weights loss: {mlp.predict(X, y)[0]:.4f}")
    mlp.fit(X, y, learning_rate=1e-4, epochs=5, batch= 10, termination_condition= 1e-3, max_iters= 100)
    predictions = mlp.predict(X, y)
    print(f"Final Loss: {predictions[0]:.4f}")
    print(f"The probabilites:{predictions[1]}")
