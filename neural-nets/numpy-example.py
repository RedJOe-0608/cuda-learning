import numpy as np
import time

# Load the binary files of the MNIST dataset. Use 10,000 training samples for faster training.
X_train = np.fromfile('./data/X_train.bin', dtype=np.float32).reshape(60000,784)[:10000]
Y_train = np.fromfile('./data/Y_train.bin',dtype=np.int32)[:10000]
X_test = np.fromfile('./data/X_test.bin',dtype=np.float32).reshape(10000,784)
Y_test = np.fromfile('./data/Y_test.bin',dtype=np.int32)

mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

#Reshape into (batch_size, channels, height, width) format
X_train = np.reshape(-1, 1, 28, 28)
X_test = np.reshape(-1, 1, 28, 28)

def relu(x):
    """
    ReLU activation function: f(x) = max(0, x)
    
    @param x: Input array
    @return: Output array with ReLU applied element-wise
    """

    return np.maximum(0,x)

def relu_derivative(x):
    """
    Derivative of the ReLU activation function.

    Returns 1.0 where x > 0, 0.0 otherwise
    
    @param x: Input array
    @return: Derivative array (1 where x > 0, 0 otherwise)
    """

    return (x > 0).astype(float)

def initialise_weights(input_size, output_size):
    """
     Initialize weights using He initialization (suitable for ReLU activations)
    Formula: weights ~ U(-sqrt(6/fan_in), sqrt(6/fan_in))
    
    @param input_size: Number of input features
    @param output_size: Number of output features
    @return: Weight matrix (input_size × output_size)
    """

    scale = np.sqrt(6.0 / input_size)

    # this gives us values between [-scale, scale]
    return (np.random.rand(input_size, output_size) * 2.0 - 1.0) * scale

def initialise_bias(output_size):
    """
    Initialise bias to zero

    @param output_size: Number of output features
    @return: Bias vector (1 x output_size)
    """

    return np.zeros((1,output_size))

     