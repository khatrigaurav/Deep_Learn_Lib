from lib.Layers import *

class CrossEntropy(Layer):
    ''' Class for Cross Entropy Loss Function'''
    def __init__(self):
        super().__init__()
        return
    
    def log_softmax(self,x):
        ''' Computes the log softmax function of x along the axis of maximum value.
        Args:
            x (numpy.ndarray): Input array of shape (num_classes, batch_size)
        Returns:
            numpy.ndarray: Output array of same shape as input array
        '''

        c = x.max(axis=0)
        logsum = np.log(np.exp(x - c).sum(axis=0))
        return x - c - logsum

    def stable_softmax(self,x):
        ''' Computes the softmax function of x along the axis of maximum value in a numerically stable way.
        Args:
            x (numpy.ndarray): Input array of shape (num_classes, batch_size)
        Returns:
            numpy.ndarray: Output array of same shape as input array
        '''
        numerator = np.exp(x-np.max(x,axis=0))
        softmax = numerator/np.sum(numerator,axis=0)
        return softmax


    def loss(self,predicted_value,true_value):
        ''' Compute crossentropy from predicted_value[batch,n_classes] and ids of correct answers
            X : output from the last layer i.e. class_num * batch_size
            y : true labels i.e. 1*batch_size '''
        
        predicted_value = self.log_softmax(predicted_value)
        loss = -np.sum(true_value*  predicted_value,axis=0)
        return np.mean(loss)
    
    def gradient(self,predicted_value,true_value):
        ''' Compute gradient of crossentropy loss with respect to predicted_value.
        Args:
            predicted_value (numpy.ndarray): Output from last layer, shape (num_classes, batch_size)
            true_value (numpy.ndarray): Ground truth labels, shape (1, batch_size)
        Returns:
            numpy.ndarray: Gradient of loss with respect to predicted_value, shape (num_classes, batch_size)
        '''

        return (self.stable_softmax(predicted_value)-true_value)/predicted_value.shape[1]


class HingeLoss(Layer):
    ''' Class for Hinge Loss Function'''
    def __init__(self) -> None:
        super().__init__()
        pass

    def loss(self,predicted_value,true_value):
        ''' Predicted Value : An array of K*num_samples , k = num of classes
            Output : A single mean value of sum across all dimensions'''
        loss = np.sum(np.maximum(0,1-predicted_value*true_value))/predicted_value.shape[1]
        return loss
    
    def gradient(self,predicted_value,true_value):
        ''' Compute gradient of hinge loss with respect to predicted_value.

        Args:
            predicted_value (numpy.ndarray): Output from last layer, shape (num_classes, batch_size)
            true_value (numpy.ndarray): Ground truth labels, shape (num_classes, batch_size)
        '''

        loss_matrix = np.zeros(predicted_value.shape)
        true_value = np.where(true_value==0,-1,true_value)
        margins = predicted_value*true_value   #element wise multiplication
        loss_matrix = np.where(margins > 0,0,margins)
        return loss_matrix
    
    def backward(self, predicted_value,true_value):
        return self.gradient(predicted_value,true_value)



class MSE(Layer):
    def __init__(self) :
        super().__init__()
        pass

    def loss(self,predicted_value,true_value):
        ''' Should return the array of losses, averaging should be done in the calling function'''
        loss = np.mean((predicted_value-true_value)**2)
        return loss
    
    def gradient(self,predicted_value,true_value):
        # assert len(predicted_value)==len(true_value), "Vectors length mismatch for gradient calculation for : Y and Yhat"
        return np.array(2*(1/max(predicted_value.shape))*np.sum(predicted_value-true_value)).reshape(-1,1)
    
    def backward(self, predicted_value, true_value):
        ''' Since it is the last layer, the backward takes only the true and predicted values'''
        return self.gradient(predicted_value,true_value)


# class Test_Loss(Layer):
#     def __init__(self) :
#         super().__init__()
#         pass

#     def loss(self,predicted_value,true_value):
#         ''' Should return the array of losses, averaging should be done in the calling function'''
#         loss = (np.square(predicted_value-true_value))/2
#         return loss
    
#     def gradient(self,predicted_value,true_value):
#         return predicted_value - true_value

#     def backward(self, predicted_value, true_value):
#         ''' Since it is the last layer, the backward takes only the true and predicted values'''
#         return self.gradient(predicted_value,true_value)

