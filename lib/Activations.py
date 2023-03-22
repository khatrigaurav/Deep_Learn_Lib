from lib.Layers import *

class RELU(Layer):
    '''
    Class for Rectified Linear Unit (ReLU) activation function layer.
    Inherits from the Layer class.
    '''
    def __init__(self) -> None:
        '''
        Initializes the RELU layer. No initializations needed.
        '''
        super().__init__()

    def forward(self,input_data):
        '''
        Performs the forward pass of the RELU layer.
        Args:
        - input: input data
        Returns:
        - Output after applying the RELU activation function.
        '''
        return np.maximum(0,input_data)
    
    def gradient(self,input):
        '''
        Calculates the gradient of the RELU activation function.

        Args:
        - input: input data , Size : features*batch_size

        Returns:
        - Gradient of the RELU activation function with respect to the input data.
        '''
        input[input<=0] = 0
        input[input > 0] = 1
        return input
    
    def backward(self, input, grad_output):
        '''
        Calculates the backward pass of the RELU layer.
        Args:
        - input: input data
        - grad_output: gradient of the output of the next layer with respect to the output of this layer
        Returns:
        - Gradient of the output of this layer with respect to the input data.
        '''

        return np.multiply(grad_output,self.gradient(input))

class Sigmoid(Layer):
    def __init__(self) -> None:
        '''
        Initializes the Sigmoid layer. No initializations needed.
        '''
        super().__init__()

    def sigmoid(self, x):
        """
        The sigmoid function maps any input value to a value between 0 and 1.
        This is useful for transforming a real-valued input to a probability score.
        
        Args:
        x: Input array , Size : features*batch_size
        
        Returns:
        The sigmoid of the input x
        """
        # Compute the sigmoid of the input x element-wise
        x[x>=0] = 1/(1+np.exp(-(x[x>=0]))) # for positive values of x
        x[x<0] = np.exp(x[x<0])/(1+np.exp(x[x<0])) # for negative values of x

        return x
    
    def forward(self, input):
        """
        Perform the forward pass for the Sigmoid activation layer.
       
         Returns:
        The output of the sigmoid activation function for the input
        """
        # Compute the output of the sigmoid activation function for the input
        return self.sigmoid(input)
    
    def gradient(self, input):
        """
        Compute the gradient of the sigmoid function with respect to its input.
        Returns:
        The gradient of the sigmoid function with respect to its input
        """
        # Compute the gradient of the sigmoid function with respect to its input element-wise
        return self.sigmoid(input) * (1 - self.sigmoid(input))
    
    def backward(self, input, grad_output):
        """
        Perform the backward pass for the Sigmoid activation layer.
        This involves computing the gradient of the loss with respect to the input.
        
        Args:
        input: Input array or scalar
        grad_output: Gradient of the loss with respect to the output of this layer
        
        Returns:
        The gradient of the loss with respect to the input of this layer
        """
        # Compute the gradient of the loss with respect to the input of this layer
        return grad_output * self.gradient(input)
