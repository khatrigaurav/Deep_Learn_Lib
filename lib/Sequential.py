# import loss_functions as Loss
# import Activations as activations
import lib.Layers as Layer


class Sequential:
    ''' This class helps us to create a sequential network architecture'''

    def __init__(self):
        self.network = []
    
    def add(self,layer):
        ''' Adds a layer to the network'''
        self.network.append(layer)
        return
    
    def add_list(self,layers_list):
        ''' Adds a list of layers to the network'''
        self.network.extend(layers_list)
        return
    
    def print_network(self):
        for layer in self.network:
            print(layer.__class__)

        return

    def construct_network(self,input_units,hidden_units,output_units, activation_function,learning_rate,weight_initialization='random'):
        ''' Helper function that constructs the network based on the input parameters
        input_units: Number of input units
        hidden_units: List of  hidden units in each layer : [256,64]
        output_units: Number of output units
        loss_function: Loss function to be used eg. loss_function.CrossEntropy,loss_function.HingeLoss
        activation_function: Activation function to be used eg. activations.Sigmoid, activations.ReLU
        '''
        self.network = []
        self.network.append(Layer.Dense(input_units,hidden_units[0],learning_rate,weight_initialization))
        self.network.append(activation_function)
        hidden_layers = len(hidden_units)-1
        
        for i in range(hidden_layers):
            self.network.append(Layer.Dense(hidden_units[i],hidden_units[i+1],learning_rate,weight_initialization))
            self.network.append(activation_function)

        self.network.append(Layer.Dense(hidden_units[-1],output_units,learning_rate,weight_initialization))
        # self.loss_function = loss_function
        
        return self.network


    def forward(self,x):
        '''
        Helper Function that runs the input through the predefined network architecture and returns the activation lists
        '''
        activations = []
        input = x   #For first layer, input is X
        network = self.network
        for layer in network:
            activations.append(layer.forward(input))
            input = activations[-1]         #latest append is the input to new layer
        
        return activations

    def train(self,X_train,y_new,loss_function):
        ''' Helper function that trains the network defined using the specified loss function
        '''
        
        activations_d = self.forward(X_train)
        logits = activations_d[-1]
    #     print(activations_d[-2].max())
        loss = loss_function.loss(logits,y_new)
        loss_grad = loss_function.gradient(logits,y_new,)

        activations_list = [X_train] + activations_d
        for reversed_layer_index in reversed(range(len(self.network))):
            layer = self.network[reversed_layer_index]
            loss_grad= layer.backward(activations_list[reversed_layer_index],loss_grad)
        
        return activations_d,loss