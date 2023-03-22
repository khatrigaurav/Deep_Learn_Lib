import numpy as np

class Layer:
    ''' Abstract Base class for MLP Layers, defines just the boilerplate for derived classes
        All the functains are defined in the derived classes.
        '''

    def __init__(self) -> None:
        pass

    def forward(self,input):
        return input
    
    def backward(self,input,grad_output):
        # dloss/dinput = dloss/dlayer * dlayer/dinput

        return np.dot(grad_output,np.eye(input.shape[1]))
    

class Dense(Layer):
    def __init__(self,input_dimension,output_dimension,learning_rate = 0.01,weight_initialization='random'):
        ''' 
        Input : Input vector , output_dimension : num of ouptputs
        Formula : Y = WX , where X is expressed in d*n and W in output* features
        Parameters:
        ----------
        input_dimension : int
            Number of input features.

        output_dimension : int
            Number of output features.

        learning_rate : float
            Learning rate for gradient descent optimization.

        weight_initialization : string
            Type of weight initialization. 'zero' or 'random'.
        '''

        self.learning_rate = learning_rate
        
        if weight_initialization == 'zero':
            self.weights = np.zeros((output_dimension,input_dimension))
            self.biases = np.zeros(output_dimension).reshape(-1,1)
            
        else:
            self.weights = np.random.randn(output_dimension,input_dimension)          #Later needs to be replaced by Kaiming Initialization
            self.biases = np.random.randn(output_dimension).reshape(-1,1)

    def forward(self, input):
        '''
        Method to perform forward pass for a given input.

        Parameters:
        ----------
        input : numpy array, shape (n, m)
            Input data to the layer.

        Returns:
        -------
        numpy array, shape (n, k)
            Result after applying forward pass on input data.

        '''
        return np.dot(self.weights,input) + self.biases
    
    def backward(self, input, grad_output):         #need to update function to check the overall weight calculations
        '''
        Method to perform backward pass for a given input and gradient output.

        Parameters:
        ----------
        input : Input data to the layer.

        grad_output : Gradient of loss with respect to output of layer.

        Returns:
        -------
        Gradient of loss with respect to input of layer.
        '''

        gradient_wrt_input = np.dot(self.weights.T,grad_output)
        grad_wrt_weight = np.dot(grad_output,input.T)
        grad_wrt_bias = grad_output.mean(axis=1,keepdims=True)
        self.weights = self.weights - self.learning_rate*grad_wrt_weight
        self.biases = self.biases - self.learning_rate*grad_wrt_bias
        return gradient_wrt_input
