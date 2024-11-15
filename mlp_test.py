import numpy as np
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
    return (a > 0).astype(float) #Returns 1 if true otherwise 0

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
    def __init__(self, initalizer, activation_function, layer_sizes):
        """
        Initialize the MLP with random weights and biases.
        :param layer_sizes: List of layer sizes. [input_dim, hidden_1, ..., hidden_k, output_dim] Example: [D, 64, 64, 11]
        """
        self.initalizer = initalizer
        self.activation_function = activation_function
        self.K = len(layer_sizes) - 1  # Number of layers excluding input
        self.weights = [self.initalizer(layer_sizes[i], layer_sizes[i + 1]) for i in range(self.K)] # Ex. w_0 (D, Unit_1), w_1 (Unit_1, Unit_2), w_2 (Unit_2, C)
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(self.K)] #https://cs231n.github.io/neural-networks-2/#datapre Recommends initalize bias as zero
        #Concatenate these two 

    
    @staticmethod
    def softmax(z):
        """Softmax activation for the output layer."""
        """NOTE: I will assume that we have the nice cross-entropy layer as described in textbook pg. 438, 
        thus not needing to calculate the diagonal jacobian of softmax."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  #Stability reasons recommended by GPT
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        """Compute the cross-entropy loss. Assume y_true is OHE."""
        """NOTE: We choose cross-entropy loss to simpify the Jacobian as described in slide chapter 9, 12.
        and textbook pg. 438"""
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m  #Stability reasons recommended by ChatGPT
    
    
    def forward(self, X):
        """
        Perform forward propagation.
        :param X: Input data of shape (n_samples, input_dim).
        :return: Activations and linear combinations for each layer.
        """
        activations = [X] #Begin at input X (N x D)
        logits = []
        for w, b in zip(self.weights[:-1], self.biases[:-1]): #Since we don't want to include the cross-entropy layer
            z = activations[-1] @ w + b #w_0 (D x Unit_1), #w_1 (Unit_1 x Unit_2), ... Thus z is always (N, Unit_i) size
            logits.append(z)
            a = self.activation_function(z)
            activations.append(a)
        
        # Output layer
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        logits.append(z)
        a = self.softmax(z)
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
        error_above = activations[-1] - y  # Output layer error (Previous layer error) Last being index -1
        weight_grads = []
        bias_grads = []
        
        for i in range(self.K-1, -1, -1):
            dW = activations[i].T @ error_above / N
            db = np.sum(error_above, axis=0, keepdims=True) / N
            weight_grads.insert(0, dW)
            bias_grads.insert(0, db)
            
            if i > 0:  # Compute error for the previous layer
                """NOTE: The Hadamard product here we use is equivalent to the diagonal derivative matrix described in the textbook."""
                error_above = (error_above @ self.weights[i].T) * self.activation_function.grad(logits[i - 1])
        
        return weight_grads, bias_grads
    
    def update_parameters(self, weight_grads, bias_grads, learning_rate):
        """
        Update weights and biases using gradients.
        :param weight_grads: Gradients for weights.
        :param bias_grads: Gradients for biases.
        :param learning_rate: Learning rate for parameter updates.
        """
        for i in range(self.K):
            self.weights[i] -= learning_rate * weight_grads[i]
            self.biases[i] -= learning_rate * bias_grads[i]
    
    def train(self, X, y, learning_rate, epochs):
        """
        Train the MLP using gradient descent.
        :param X: Input data.
        :param y: True labels (one-hot encoded).
        :param learning_rate: Learning rate.
        :param epochs: Number of training iterations.
        """
        for epoch in range(epochs):
            activations, logits = self.forward(X)
            loss = self.cross_entropy_loss(y, activations[-1])
            weight_grads, bias_grads = self.backward(X, y, activations, logits)
            self.update_parameters(weight_grads, bias_grads, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")


if __name__ == "__main__":
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Generate dummy data
    X = np.random.rand(100, 3)  # 100 samples, 3 features
    y_raw = np.random.randint(0, 3, 100)  # Random integer labels for 3 classes
    y = np.eye(3)[y_raw]  # Convert to one-hot encoding
    
    # Initialize and train the MLP
    activation_function = ActivationFunction(func = ReLU, derivative= ReLU_derivative)
    mlp = MLPSoftmax(initalizer= random_initalizer, activation_function= activation_function, layer_sizes= [3, 10, 15, 11])  # Input: 3, Hidden: 5, Output: 3 (softmax)
    mlp.train(X, y, learning_rate=0.1, epochs=1000)
