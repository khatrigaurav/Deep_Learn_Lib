from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
import numpy as np
import os


class DataLoader():
    def __init__(self,batch_size=256,download=False):
        self.batch_size = batch_size
        data_path = os.path.join(os.getcwd(),'data')
        
        if download==True:
            self.X, self.y = datasets.fetch_openml( "mnist_784", version=1, return_X_y=True, as_frame=False)
            self.X = self.X/255
            
            with open(os.path.join(data_path,'X.npy'),'wb') as f:
                np.save(f,self.X)

            with open(os.path.join(data_path,'y.npy'),'wb') as f:
                np.save(f,self.y)

        else:
            self.X = np.load(os.path.join(data_path,'X.npy'),allow_pickle=True)
            self.y = np.load(os.path.join(data_path,'y.npy'),allow_pickle=True)

        return
    
    def train_test_gen(self,split_ratio = 0.75):
        ''' Divides the given dataset into train and a test set'''
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,train_size=split_ratio,random_state=101)

        return X_train,X_test,y_train,y_test
    
    def batch_generator(self,X,y):
        '''Splits the entire X,y dataset into batches'''
        # master_indexes = np.random.permutation(len(X))
        for i in range(0,X.shape[1],self.batch_size):
            # yield X[:,i:i+batch_size]
            # yield y[:,i:i+batch_size]
            yield i


