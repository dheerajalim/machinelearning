import numpy as np


class Knnmodel:

    def __init__(self, train_file, test_file, _kvalue):
        """

        :param train_file: The filename for the training instance
        :param test_file:  The filename  for  the test instance
        :param _kvalue: The K value for the KNN model

        """
        self.train_dataset = np.genfromtxt(train_file, delimiter=',')  # Reads the training dataset
        self.test_dataset = np.genfromtxt(test_file, delimiter=',')  # Reads the test Dataset
        self.k_value = _kvalue  # Sets the user input k value
        self.train_data = np.empty  # An empty numpy array to store train data features
        self.train_class = np.empty  # An empty numpy array to store train data class
        self.test_data = np.empty  # An empty numpy array to store test data features
        self.test_class = np.empty  # An empty numpy array to store test data class

    def dataset(self, class_column):
        """
        Filters the dataset to store the features and classes in separate numpy arrays
        :param class_column: The number of feature columns
        :return: None
        """

        self.train_data = np.delete(self.train_dataset, class_column, axis=1)  # Contains the training features
        self.train_class = self.train_dataset[:, class_column]  # Contains the training class

        self.test_data = np.delete(self.test_dataset, class_column, axis=1)  # Contains the test features
        self.test_class = self.test_dataset[:, class_column]  # Contains the test class

    def calculateDistances(self, query_instance, feature_list):
        """

        :param query_instance: Contains the single instance of the test data (1D Numpy Array)
        :param feature_list:  Contains all the features of the training data (2D Numpy array)
        :return: Euclidean distance between the training and query Instances , Sorted Indices Array
        """
        feature_difference = feature_list - query_instance
        euclidena_distance = np.sqrt(np.sum(np.square(feature_difference), axis=1))  # Euclidean distance calculation
        sorted_distance_index = np.argsort(euclidena_distance)  # Sorted array of distance indices
        return euclidena_distance, sorted_distance_index

    @staticmethod
    def knn_vote(prediction):
        """
        :param prediction: The list of all K nearest neighbours
        :return: Mode of the k nearest Neighbours
        """
        return np.bincount(prediction).argmax()

    def basic_knn_percentage(self, results):
        """
        Calculates the accuracy of Basic KNN Model
        :param results: Numpy array of Euclidean Distances and Sorted Distances
        :return: Accuracy of the model
        """
        sorted_indices = results[:, 1].astype('int32')  # Array of sorted indices based on euclidean weights
        k_nearest = sorted_indices[:, :self.k_value]  # Selection of K indices from the sorted_indices
        # Numpy array to store the classes predicted for the test data
        prediction = self.train_class[k_nearest].astype('int32')
        # Finding the mode of the classes in K neighbours
        find_res = np.apply_along_axis(self.knn_vote, 1, prediction)
        # Calculating the count of correct predictions
        correct_prediction = np.count_nonzero(self.test_class == find_res)
        # The percentage of correct prediction
        percentage = (correct_prediction / len(self.test_dataset)) * 100
        return percentage

    def weighted_knn_inverse_distance(self, prediction, distances_k_nearest, n_value=1):
        """

        :param n_value: Contains the value of n for the inverse power calculation
        :param prediction: K Predicted classes of the test dataset
        :param distances_k_nearest: Euclidean Distances of the K predicted classes
        :return: Numpy array of predicted classes on test data after inverse distance calculation
        """
        find_res = np.zeros(shape=(1000,))  # Creating an empty numpy array of size 1000 to store the predicted class

        for i in range(0, len(prediction)):
            class0 = 0
            class1 = 0
            class2 = 0

            for j in range(0, self.k_value):
                if prediction[i][j] == 0:
                    class0 += (1 / pow(distances_k_nearest[i][j], n_value))
                elif prediction[i][j] == 1:
                    class1 += (1 / pow(distances_k_nearest[i][j], n_value))
                elif prediction[i][j] == 2:
                    class2 += (1 / pow(distances_k_nearest[i][j], n_value))

            if class0 > class1 and class0 > class2:
                find_res[i] = 0

            elif class1 > class0 and class1 > class2:
                find_res[i] = 1

            elif class2 > class0 and class2 > class1:
                find_res[i] = 2

        return find_res

    def weighted_knn_percentage(self, results, n_value):

        """
        Calculates the accuracy of weighted KNN model
        :param n_value: Contains the value of n for the inverse power calculation
        :param results: Numpy array of Euclidean Distances and Sorted Distances
        :return: Accuracy of the model
        """

        sorted_indicies = results[:, 1].astype('int32')  # Array of sorted indices based on euclidean weights
        k_nearest = sorted_indicies[:, :self.k_value]   # Selection of K indices from the sorted_indices
        """
        Sorting the Euclidean distance array and finding the K distances from that
        """
        distances_original = results[:, 0]
        distances = distances_original.copy()
        distances.sort(axis=1)
        distances_k_nearest = distances[:, :self.k_value]

        # Numpy array to store the classes predicted for the test data
        prediction = self.train_class[k_nearest].astype('int32')
        # Finding the Inverse of distance of the classes in K neighbours
        find_res = self.weighted_knn_inverse_distance(prediction, distances_k_nearest, n_value)
        # Calculating the count of correct predictions
        correct_prediction = np.count_nonzero(self.test_class == find_res)
        # The percentage of correct prediction
        percentage = (correct_prediction / len(self.test_dataset)) * 100
        return percentage
