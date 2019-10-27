import numpy as np
import matplotlib.pyplot as plt


class InvalidKValue(Exception):
    def __init__(self, message):
        self.message = message;


class Knnmodel:

    def __init__(self, train_file, test_file):
        """

        :param train_file: The filename for the training instance
        :param test_file:  The filename  for  the test instance
        :param _kvalue: The K value for the KNN model

        """
        self.train_dataset = np.genfromtxt(train_file, delimiter=',')  # Reads the training dataset
        self.test_dataset = np.genfromtxt(test_file, delimiter=',')  # Reads the test Dataset
        # self.k_value = _kvalue  # Sets the user input k value
        self.train_data = np.empty  # An empty numpy array to store train data features
        self.train_class = np.empty  # An empty numpy array to store train data class
        self.test_data = np.empty  # An empty numpy array to store test data features
        self.test_class = np.empty  # An empty numpy array to store test data class
        self.scaled_train_data = np.empty   # An empty numpy array to store scaled train data
        self.scaled_test_data = np.empty    # An empty numpy array to store scaled test data

    def dataset(self, class_column):
        """
        Filters the dataset to store the features and classes in separate numpy arrays
        :param class_column: The number of feature columns
        :return: None
        """
        print('Building KNN Model...')
        self.train_data = np.delete(self.train_dataset, class_column, axis=1)  # Contains the training features
        self.train_class = self.train_dataset[:, class_column]  # Contains the training class

        self.test_data = np.delete(self.test_dataset, class_column, axis=1)  # Contains the test features
        self.test_class = self.test_dataset[:, class_column]  # Contains the test class

    def dataset_scaling(self):
        """

        :return: Returns the scaled version of the dataset
        """
        print('Scaling Dataset...')
        scaling_train_data = self.train_data.copy()
        scaling_test_data = self.test_data.copy()
        min_features_train = np.amin(scaling_train_data, axis=0)
        max_features_train = np.amax(scaling_train_data, axis=0)
        self.scaled_train_data = (scaling_train_data - min_features_train) / (max_features_train - min_features_train)
        self.scaled_test_data = (scaling_test_data - min_features_train) / (max_features_train - min_features_train)

    def manhattanDistance(self, query_instance, feature_list):
        """

        :param query_instance: Contains the single instance of the test data (1D Numpy Array)
        :param feature_list:  Contains all the features of the training data (2D Numpy array)
        :return: Manhatan distance between the training and query Instances , Sorted Indices Array
        """
        feature_difference = feature_list - query_instance
        manhattan_distance = np.sum(np.absolute(feature_difference), axis=1) # Manhattan Distance Calculation
        sorted_distance_index = np.argsort(manhattan_distance)  # Sorted array of distance indices
        return manhattan_distance, sorted_distance_index

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

    def basic_knn_percentage(self, results, k_value):
        """
        Calculates the accuracy of Basic KNN Model
        :param k_value:
        :param results: Numpy array of Euclidean Distances and Sorted Distances
        :return: Accuracy of the model
        """

        sorted_indices = results[:, 1].astype('int32')  # Array of sorted indices based on euclidean weights
        k_nearest = sorted_indices[:, :k_value]  # Selection of K indices from the sorted_indices
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
        find_res = np.zeros(shape=(len(self.test_data),))  # Creating an empty numpy array of size 1000 to store the predicted class
        unique_classes = np.unique(prediction)
        distance_inverse = 1 / pow(distances_k_nearest, n_value)
        for i in range(0, len(prediction)):
            classification_list = []
            for j in unique_classes:
                classes = np.where(prediction[i] == j)
                sum_inverse_distance = distance_inverse[i][classes]
                classification_list.append(np.sum(sum_inverse_distance))

            find_res[i] = classification_list.index(max(classification_list))

        return find_res

    def weighted_knn_percentage(self, results, k_value, n_value):

        """
        Calculates the accuracy of weighted KNN model
        :param k_value:
        :param n_value: Contains the value of n for the inverse power calculation
        :param results: Numpy array of Euclidean Distances and Sorted Distances
        :return: Accuracy of the model
        """
        sorted_indicies = results[:, 1].astype('int32')  # Array of sorted indices based on euclidean weights
        k_nearest = sorted_indicies[:, :k_value]   # Selection of K indices from the sorted_indices
        """
        Sorting the Euclidean distance array and finding the K distances from that
        """
        distances_original = results[:, 0]
        distances = distances_original.copy()
        distances.sort(axis=1)
        distances_k_nearest = distances[:, :k_value]

        # Numpy array to store the classes predicted for the test data
        prediction = self.train_class[k_nearest].astype('int32')
        # Finding the Inverse of distance of the classes in K neighbours
        find_res = self.weighted_knn_inverse_distance(prediction, distances_k_nearest, n_value)
        # Calculating the count of correct predictions
        correct_prediction = np.count_nonzero(self.test_class == find_res)
        # The percentage of correct prediction
        percentage = (correct_prediction / len(self.test_dataset)) * 100
        return percentage

    def weighted_regression_knn_percentage(self, results, k_value, n_value):

        """
        Calculates the accuracy of weighted KNN model
        :param k_value:
        :param n_value: Contains the value of n for the inverse power calculation
        :param results: Numpy array of Euclidean Distances and Sorted Distances
        :return: Accuracy of the model
        """

        sorted_indicies = results[:, 1].astype('int32')  # Array of sorted indices based on euclidean weights
        k_nearest = sorted_indicies[:, :k_value]   # Selection of K indices from the sorted_indices
        """
        Sorting the Euclidean distance array and finding the K distances from that
        """
        distances_original = results[:, 0]
        distances = distances_original.copy()
        distances.sort(axis=1)
        distances_k_nearest = distances[:, :k_value]

        # Numpy array to store the regression values predicted for the test data
        prediction = self.train_class[k_nearest]
        # Calculation of the distance weighted regression values
        try:
            find_res = np.divide(np.sum(np.multiply(1 / pow(distances_k_nearest, n_value), prediction), axis=1),
                                 np.sum(1 / pow(distances_k_nearest, n_value), axis=1))

        except (ZeroDivisionError,ValueError, TypeError) as e:
            print(f'Unable to calculate the predictions , error : {e}')


        """The R 2 coefficient calculation"""
        r_square = self.r_square_coefficient(find_res)
        percentage = r_square * 100
        return percentage

    def r_square_coefficient(self,find_res):
        # Sum of squared residuals. The numerator
        ssr = np.sum(np.square(np.subtract(find_res, self.test_class)))
        # Total sum of squares. The Denominator
        sst = np.sum(np.square(np.subtract(self.test_class, np.average(self.test_class))))
        # R square coefficient value
        r_square = 1-(ssr/sst)

        return r_square


class PlotGraph:

    def __init__(self):
        pass

    @staticmethod
    def plot_graph(k_graph_values, accuracy_graph_values):
        plt.plot(k_graph_values, accuracy_graph_values,marker='o')

    @staticmethod
    def show_graph(legend = [], filename='Dummy_KNN', n_value = 1):
        plt.ylabel('Accuracy Percentage')
        plt.xlabel('Value of K')
        plt.legend(legend)
        if n_value != 1:
            plt.title(f'n = {n_value}')
        plt.savefig(filename)

        plt.show()
