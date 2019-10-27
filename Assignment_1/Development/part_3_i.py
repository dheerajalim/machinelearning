"""
Author: Dheeraj Alimchandani
Student ID : R00182505
"""

from knn_model import *


class WeightedRegressionKnn:

    def __init__(self, train_file, test_file, _plotgraph=False):
        """
        :param train_file: The filename for the training instance
        :param test_file:  The filename  for  the test instance
        :param _plotgraph:  Boolean for plotting graph

        """

        self.knn_model = Knnmodel(train_file, test_file)    # Initializing KNN Model
        self.knn_model.dataset(12)  # Creating the Dataset
        # Calculating the Distances
        self.results = np.apply_along_axis(self.knn_model.calculateDistances, 1, self.knn_model.test_data,
                                           self.knn_model.train_data)
        self.accuracy_graph_values = []
        self.k_graph_values = []
        self.plot_graph = _plotgraph

    def prediction(self, k_value=1, n_value=1):
        """
        Calculates the euclidean distance between each query instance and the train dataset and returns accuracy
        prediction
        :param k_value: k nearest neighbours
        :param n_value: Contains the value of n for the inverse power calculation
        :return:  Accuracy of the prediction
        """
        try:
            if k_value < 1 :
                raise InvalidKValue(k_value)

        except InvalidKValue as e:
            print(f'Invalid neighbour value: {e.message} ')
            return

        try:
            percentage = self.knn_model.weighted_regression_knn_percentage(self.results, k_value, n_value)
            print(f'The Weighted KNN Regression model with k = {k_value} and n = {n_value}, '
                  f'has an accuracy of {round(percentage, 2)} %')

            if self.plot_graph:
                self.accuracy_graph_values.append(round(percentage, 2))
                self.k_graph_values.append(k_value)

        except Exception as e:
            print(f'Error finding accuracy for K = {k_value}, error {e}')


if __name__ == '__main__':
    # Initializing the  weighted_regression_knn Model
    weighted_regression_knn = WeightedRegressionKnn(Parameters.TRAIN_DATA_REGRESSION,
                                                    Parameters.TEST_DATA_REGRESSION,
                                                    _plotgraph=Parameters.PLOT_GRAPH)
    for k in range(1, Parameters.LIMIT + 1):
        weighted_regression_knn.prediction(k, Parameters.n)

    PlotGraph.plot_graph(weighted_regression_knn.k_graph_values, weighted_regression_knn.accuracy_graph_values)
    PlotGraph.show_graph(Parameters.LEGEND, filename='Regression_graph', n_value= Parameters.n)
