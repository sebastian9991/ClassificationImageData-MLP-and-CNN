import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from typing import Literal, Optional
import time
from tabulate import tabulate
import psutil 
import os
"""Plotting functions memory/ File related variables."""
def get_memory_usage():
    """This function is meant to display the memory usage in the 3.5"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024) #MB

def log_memory_usage(time, filename = "memory_log.txt"):
    with open(filename, "a") as f:
        f.write(f"Memory usage at {time}: {get_memory_usage()} MB\n")

def remove_memory_log():
    try:
        os.remove("memory_log.txt")
    except FileNotFoundError:
        print("memory_log.txt not found")

save_figure_path = "./figs/"
        

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

def sigmoid_derivative(a):
    """Derivative of Sigmoid activation."""
    return sigmoid(a) * (1 - sigmoid(a))

def tanh(z):
    """Tanh activation function."""
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def tanh_derivative(a):
    """Derivative of Tanh"""
    return 1 - np.power(tanh(a), 2)

def leaky_ReLU(z, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(z > 0, z, alpha*z)
def leaky_ReLU_derivative(a, alpha = 0.01):
    """Derivative of Leaky ReLU."""
    return np.where(a > 0, 1.0, alpha)

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


    def func(self, x):
        return self.func(x)

    def grad(self, x):
        return self.derivative(x)
"""
Guidance from: https://medium.com/@andresberejnoi/how-to-implement-backpropagation-with-numpy-andres-berejnoi-e7c14f2e683ac
and: https://zerowithdot.com/mlp-backpropagation/

"""
class MLPSoftmax:
    """
    MLP Softmax class
    """
    def __init__(self, initalizer, activation_functions, layer_sizes, normalization = True):
        """
        Initialize the MLP with random weights and biases.
        :param layer_sizes: List of layer sizes. [input_dim, hidden_1, ..., hidden_k, output_dim] Example: [D, 64, 64, 11]
        """
        self.initalizer = initalizer #Defined above we use different initalizers depending on the activation function
        self.normalization = normalization
        self.K = len(layer_sizes) - 1  # Number of layers excluding input
        """Usage of dictionaries guided from: https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795"""
        self.activation_functions = {}
        self.weights = {}
        self.biases = {}
        self.weights.update({i + 1: self.initalizer(size_in, size_out) for i, (size_in, size_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))})
        self.biases.update({i + 1: np.zeros((1, size_out)) for i, size_out in enumerate(layer_sizes[1:])})
        self.activation_functions.update({i + 2: activation_functions[i] for i in range(self.K)})


    
    
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
        if self.use_batch_normalization == True:
            mean = np.mean(a, axis=0, keepdims=True)
            variance = np.var(a, axis=0, keepdims=True)
            x_norm = (a - mean) / np.sqrt(variance + eps)
            return gamma * x_norm + beta
        else:
            return a
    
    def clip_gradients(grads, clip_value = 5):
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
        activations = {} 
        logits = {}
 
        activations[1] = X #Begin at input X (N x D)
        for i in range(1, self.K + 1):
            logits[i + 1] = activations[i] @ self.weights[i] + self.biases[i]#w_0 (D x Unit_1), #w_1 (Unit_1 x Unit_2), ... Thus z is always (N, Unit_i) size
            activations[i + 1] = self.batch_normalization(self.activation_functions[i + 1].func(logits[i + 1])) #Use batch normalizer here
        return activations, logits
    
    """Guidance from backpropgation MLP section in: https://www.kaggle.com/code/vitorgamalemos/multilayer-perceptron-from-scratch?scriptVersionId=79241565
        and from Textbook Kevin P. Murphy PML, pg. 438."""
    def backward(self, X, y, activations, logits):
        """
        Perform backward propagation.
        :param X: Input data.
        :param y: True labels (one-hot encoded).
        :param activations: Activations from the forward pass.
        :param Logits: Linear combinations from the forward pass.
        :return: Gradients for weights and biases.
        """
        """Formula guidance here: https://github.com/KirillShmilovich/MLP-Neural-Network-From-Scratch/blob/master/MLP.ipynb"""
        N = X.shape[0]
        error_above = (activations[self.K + 1] - y)*(self.activation_functions[self.K + 1].grad(activations[self.K + 1]))  # Output layer error (Previous layer error) our K. Last being index -1 DIM(N, Unit_i)
        dW = activations[self.K].T @ error_above
        db = np.mean(error_above, axis= 0)
        weight_bias_pair = {
            self.K: (dW, db)
        }
        
        for i in range(self.K, 1, -1): # i = K + 1, ..., 1
            """NOTE: The Hadamard product here we use is equivalent to the diagonal derivative matrix described in the textbook. pg.438"""
            error_above = (error_above @ self.weights[i].T) * self.activation_functions[i].grad(logits[i]) # ((N, Unit_i) (Unit_{i - 1}, Unit_i)T) * (N, Unit_{i - 1}) NOTE: i - 1 because logits contains the output logit
            dW = activations[i - 1].T @ error_above # (N, Unit_i)T @ (N, Unit_i) => (Unit_i, N) @ (N, Unit_i)
            db = np.mean(error_above, axis=0) #May remove this in favor of biases in weights matrices
            weight_bias_pair[i - 1] = (dW, db)
            # print(f"Layer {i}, Max Weight error: {np.max(error_above)}, Min Weight error: {np.min(error_above)}")
            # print(f"Layer {i}, Max Weight Grad: {np.max(dW)}, Min Weight Grad: {np.min(dW)}")

            
        for layer, weights_bias in weight_bias_pair.items():
            self.update_parameters(layer, weight_grads=weights_bias[0], bias_grads=weights_bias[1])
        
    
    def update_parameters(self, layer_index, weight_grads, bias_grads):
        """
        Update weights and biases using gradients.
        :param weight_grads: Gradients for weights.
        :param bias_grads: Gradients for biases.
        :param learning_rate: Initial learning rate for parameter updates.
        :param iterations: Current iteration for the scheduler. 
        :param scheduler_p: Parameter for the scheduler
        """
        #learning_rate = learning_rate / (1 + t ** scheduler_p) #Schedule just made it diminish too quickly

        if(self.regularization != None):
            if(self.regularization == 'l1'):
                weight_grads += self.lambbda*self.weights[layer_index]
            elif(self.regularization == 'l2'):
                weight_grads += self.lambbda*np.sign(self.weights[layer_index])



        self.weights[layer_index] = self.weights[layer_index] - self.learning_rate * weight_grads
        #print(f"difference of weights to grads: {np.linalg.norm(self.weights[layer_index]) - np.linalg.norm(weight_grads)}")
        self.biases[layer_index] = self.biases[layer_index] - self.learning_rate * bias_grads

        grad_norm = np.linalg.norm(np.hstack([g.ravel() for g in weight_grads + bias_grads])) #Compute norm over all our gradients
        if (grad_norm < self.termination_condition):
            return True
        
        return False
    
    def fit(self, X_train, y_train, X_test, y_test, learning_rate, epochs, batch,  termination_condition, max_iters, save_figure_name = "plot", architecture_title = "",  display_plot = True, use_batch_normalization = True, lambbda = 0, regularization: Optional[Literal['l1', 'l2']] = None) :
        """
        Train the MLP using gradient descent.
        :param X: Input data.
        :param y: True labels (one-hot encoded).
        :param learning_rate: Learning rate.
        :param epochs: Number of training iterations.
        """
        N, D = X_train.shape
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.use_batch_normalization = use_batch_normalization
        self.termination_condition = termination_condition
        if regularization != None:
            assert(lambbda != 0)
            self.lambbda = lambbda
        train = []
        test = []
        epoch_time = []
        remove_memory_log()
        try:
            for epoch in range(epochs):
                start_time = time.time()
                iterations = 0
                ##np.shuffle with indices works much better than np.random.choice
                N = X_train.shape[0]
                shuffled_indices = np.random.permutation(N)
                x_shuffled = X_train[shuffled_indices]
                y_shuffled = y_train[shuffled_indices]
                for start_idx in range(0, N, batch):
                    end_idx = min(start_idx + batch, N)
                    batch_x = x_shuffled[start_idx:end_idx]
                    batch_y = y_shuffled[start_idx:end_idx]

                    activations, logits = self.forward(batch_x) #Make sure the batch iterates. I did not do this in my second assignment, without it the output is volatile like single point SGD
                    stop_ = self.backward(batch_x, batch_y, activations, logits)

                    if stop_ and iterations >= max_iters: #Conditional Check for terminations 
                        break

                end_time = time.time()
                log_memory_usage(time=start_time)
                epoch_time.append(end_time - start_time)
                train.append(self.evaluate_acc(np.argmax(y_train, axis= 1), self.predict(X_train)))
                test.append(self.evaluate_acc(np.argmax(y_test, axis=1), self.predict(X_test)))
                #Plot within the function
                if display_plot:
                    clear_output()
                    print("Epoch", epoch)
                    print("Final Train accuracy:", train[-1])
                    print("Final Test accuracy:", test[-1])
                    plt.plot(train, label = 'train accuracy')
                    plt.plot(test, label='test accuracy')
                    plt.legend(loc = 'best')
                    plt.ylabel('Accuracy')
                    plt.xlabel('epochs')
                    plt.title("Accuracy over epochs, architecture: " + str(architecture_title))
                    plt.grid()
                    if epoch == epochs - 1:
                        print("Saving Image")
                        plot_filename = os.path.join(save_figure_path, save_figure_name + ".png")
                        plt.savefig(plot_filename)
                    plt.show()
        except Exception as e:
            log_memory_usage(time = 0)
            print("An error occured: ", e) 
        

        total_time = np.mean(epoch_time)
        formatted_total_time = f"{total_time:.4f}"
        table = [[formatted_total_time]]
        print(tabulate(table, headers=["Average Time per epoch (s)"], tablefmt="pretty"))
    
    def predict(self, X):
        """
        Perform forward propagation.
        :param X: Input data of shape (n_samples, input_dim).
        :param y: True labels. (n_samples, Classes)
        :return: loss / or actual label
        """
        activations, _ = self.forward(X)
        output_probabilites = activations[self.K + 1] #Softmax layer
        return np.argmax(output_probabilites, axis=1) #Assuming that the index corresponds to the class OHE


    def evaluate_acc (self, y_true, y_pred):
        """
        Our evaluation function based on the amount of equal labels 
        (To interpret evaluation better than the loss function.)
        """
        return np.sum(y_true == y_pred)/y_pred.shape[0]



if __name__ == "__main__":
    ##TESTING ON MAIN FUNCTION
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
    mlp = MLPSoftmax(initalizer= random_optimized_initalizer, activation_functions= [loss_activation_function, activation_function_relu], layer_sizes= [13, 256, 3])  # Input: 3, Hidden: 5, Output: 3 (softmax)
    mlp.fit(X_train=X, y_train= y,X_test=X_test, y_test=y_test, learning_rate=1e-4, epochs=50, batch= 10, termination_condition= 1e-3, max_iters= 1000, display_plot=True)
    predictions = mlp.predict(X)
    print(f"Final Loss: {predictions}")
    print(f"True labels: {y_list}")
    print(f"Final Accuracy:{mlp.evaluate_acc(y_true = y_list, y_pred= predictions)}")
