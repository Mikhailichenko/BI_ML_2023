import numpy as np
import statistics

class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use
        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops
        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        distances = np.zeros((X.shape[0], self.train_X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(self.train_X.shape[0]):
                distances[i][j] = sum(abs(X[i] - self.train_X[j]))
        return distances
    

    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used
        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        
        distances = np.zeros((X.shape[0], self.train_X.shape[0]))
        for i in range(X.shape[0]):
            distances[i] = abs(X[i] - self.train_X).sum(1)
        return distances


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy
        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        distances = np.sum(abs(X[:, np.newaxis] - self.train_X[np.newaxis, :]), axis = -1)
        return distances


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)

        def find_max_mode(list1):
            list_table = statistics._counts(list1)
            len_table = len(list_table)
    
            if len_table == 1:
                max_mode = statistics.mode(list1)
            else:
                new_list = []
                for i in range(len_table):
                    new_list.append(list_table[i][0])
                max_mode = max(new_list) 
            return max_mode
        
        
        for i in range(n_test):
            if self.k == 1:
                prediction[i] = self.train_y[distances[i].argmin()]
            
            else:
                k_sorted_index = distances[i].argsort()[:self.k]
                y_new = []

                for j in k_sorted_index:
                    y_new.append(self.train_y[j])
                
                prediction[i] = find_max_mode(y_new)
        return prediction

       


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, np.int)

        
        def find_max_mode(list1):
            list_table = statistics._counts(list1)
            len_table = len(list_table)
    
            if len_table == 1:
                max_mode = statistics.mode(list1)
            else:
                new_list = []
                for i in range(len_table):
                    new_list.append(list_table[i][0])
                max_mode = max(new_list) 
            return max_mode
        
        for i in range(n_test):
            if self.k == 1:
                prediction[i] = self.train_y[distances[i].argmin()]
            
            else:
                k_sorted_index = distances[i].argsort()[:self.k]
                y_new = []

                for j in k_sorted_index:
                    y_new.append(self.train_y[j])
                
                prediction[i] = find_max_mode(y_new)
        return prediction
