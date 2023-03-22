"""
Created on March 4 12:00:00 2020

@author: Gaurav
@version: 1.0

Helper function for data preprocessing, accuracy measurment, and data batch generation
"""
import numpy as np
import matplotlib.pyplot as plt
from lib.DataLoader import DataLoader

def process_data(download=False,ratio=0.75):
    ''' This functions downloads the data, splits into train-test sets and creates one hot 
        encodings of categorical outputs
    '''
    dataloader = DataLoader(download=download)
    X_train,X_test,y_train,y_test = dataloader.train_test_gen(split_ratio=ratio)
    y_train = y_train.astype('int')
    y_test  = y_test.astype('int')
    X_train = (X_train).T
    y_train = (y_train.reshape(1,-1))
    y_test = (y_test.reshape(1,-1))
    
    return X_train,X_test,y_train,y_test

def get_accuracy(predicted_labels,true_labels,identifier):
    ''' A function to compute accuracy of predicted labels 
    Args:
        predicted_labels (numpy.ndarray): An array of predicted labels , dimension :  classes*samples
        true_labels (numpy.ndarray)     : An array of true labels , dimension :  classes*samples
        identifier (str)                : A string indicating whether the accuracy is for 'train' or 'test' data.

    Returns:
        float: The accuracy score as a value between 0 and 1.
    '''

    if identifier =='train':
        accuracy_score = np.mean(np.argmax(predicted_labels,axis=0).reshape(1,-1) ==np.argmax(true_labels,axis =0).reshape(1,-1))
    
    if identifier == 'test':
        accuracy_score = np.mean(np.argmax(predicted_labels,axis=0).reshape(1,-1) == true_labels)
    
    return accuracy_score


def main(network,loss_function,data_dict,batch_size=512,epoch_range = 80):
    """Train a neural network and report the training and validation accuracies and losses.

    Args:
        network (object)        : An object representing the neural network.
        loss_function (object)  : An object representing the loss function to use for training.
        data_dict (dict)        : A dictionary containing the training and testing data.
        batch_size (int)        : The number of samples per batch.
        epoch_range (int)       : The number of epochs to train for.

    Returns:
        tuple: A tuple containing:
            - **train_accuracies** (*list*) - A list of training accuracies.
            - **test_accuracies** (*list*) - A list of testing accuracies.
            - **epoch_losses** (*list*) - A list of epoch losses.
    """
    dataloader = DataLoader(batch_size)
    train_accuracies = []
    test_accuracies = []
    loss_measured = []
    epoch_losses = []

    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train_new = data_dict['y_train_new']
    y_test = data_dict['y_test']

    for epoch in range(epoch_range):
        epoch_train_accuracies = []

        db = dataloader.batch_generator(X_train,y_train_new)
        for batch_index in db:
            batch_X = X_train[:,batch_index:batch_index+dataloader.batch_size]
            batch_y = y_train_new[:,batch_index:batch_index+dataloader.batch_size]
            activations_d,loss = network.train(batch_X,batch_y,loss_function)
            loss_measured.append(loss)
            accuracy_ = get_accuracy(activations_d[-1],batch_y,'train')
            epoch_train_accuracies.append(accuracy_)

        #For test accuracy
        test_data_activation = network.forward(X_test.T)[-1]
        test_accuracy = get_accuracy(test_data_activation,y_test,'test')
        test_accuracies.append(test_accuracy)

        train_accuracies.append(np.mean(epoch_train_accuracies))
        epoch_losses.append(np.mean(loss_measured))


        print(f'Iterating Epoch {epoch+1}/{epoch_range},   Average Epoch Loss  : {round(np.mean(loss_measured),6)},Training Accuracy : {round(np.mean(epoch_train_accuracies),6)}, Validation Accuracy : {round(test_accuracy,6)}  ')
    
    return train_accuracies, test_accuracies, epoch_losses

def _plot(train_accuracies,test_accuracies,loss_measured):
    fig = plt.figure(figsize=(12, 6))

    plt.subplots_adjust(wspace= 0.25, hspace= 0.25)

    sub1 = fig.add_subplot(1,2,1) # two rows, two columns, fist cell
    # plt.annotate(xy = (0.5, 0.5), va = 'center', ha = 'center',  weight='bold', fontsize = 15)
    plt.plot(range(1,1+len(train_accuracies)),train_accuracies,label = 'Train')
    plt.plot(range(1,1+len(test_accuracies)),test_accuracies, label = 'Test')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy',fontsize= 12)
    plt.grid()
    plt.legend()


    # Create second axes, the top-left plot with orange plot
    sub2 = fig.add_subplot(1,2,2) # two rows, two columns, second cell
    # plt.annotate('sub2', xy = (0.5, 0.5), va = 'center', ha = 'center',  weight='bold', fontsize = 15)
    plt.plot(range(1,1+len(loss_measured)),loss_measured,linestyle = 'dashed',color='red', label = 'Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss',fontsize= 12)
    plt.legend()
    plt.grid()

    plt.show()
