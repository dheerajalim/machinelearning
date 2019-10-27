"""
Comprehensive range of techniques :
1 . Using scaling on the KNN model to see the improvement in results

"""

from knn_model import *
from part_1_oop import BasicKnn
from part_2_a import WeightedKnn


class ScaledKnn:

    def __init__(self, train_file, test_file, _plotgraph=False):
        """
        :param train_file: The filename for the training instance
        :param test_file:  The filename  for  the test instance
        :param _kvalue: The K value for the KNN model

        """
        self.knn_model = Knnmodel(train_file, test_file)
        self.knn_model.dataset(10)
        self.knn_model.dataset_scaling()

        self.results = np.apply_along_axis(self.knn_model.calculateDistances, 1, self.knn_model.scaled_test_data,
                                           self.knn_model.scaled_train_data)
        self.accuracy_graph_values = []
        self.k_graph_values = []
        self.plot_graph = _plotgraph

    def prediction(self, k_value=1, n_value=1, type='basic'):
        """
        Calculates the euclidean distance between each query instance and the train dataset and returns accuracy
        prediction
        :return:  Accuracy of the prediction
        """
        try:
            if k_value < 1:
                raise InvalidKValue(k_value)
        except InvalidKValue as e:
            print(f'Invalid neighbour value: {e.message} ')
            return

        try:
            if type == 'basic':
                percentage = self.knn_model.basic_knn_percentage(self.results, k_value)
                print(f'The Scaled Basic KNN model with k = {k_value}, has and accuracy of {round(percentage, 2)} %')
            elif type == 'weighted':
                percentage = self.knn_model.weighted_knn_percentage(self.results, k_value, n_value)
                print(f'The Scaled Weighted KNN model with k = {k_value} and n = {n_value},'
                      f' has and accuracy of {round(percentage, 2)} %')

            if self.plot_graph:
                self.accuracy_graph_values.append(round(percentage, 2))
                self.k_graph_values.append(k_value)

        except Exception as e:
            print(f'Error finding accuracy for K = {k_value}, error {e}')

    def clean_graph_values(self):
        self.accuracy_graph_values = []
        self.k_graph_values = []


if __name__ == '__main__':
    PLOT_GRAPH = True

    LEGEND = []
    LIMIT = 20
    n = 2
    scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _plotgraph=PLOT_GRAPH)
    for k in range(1, LIMIT + 1):
        scaled_knn.prediction(k, type='basic')
    PlotGraph.plot_graph(scaled_knn.k_graph_values, scaled_knn.accuracy_graph_values)
    LEGEND.append('Basic Scaled KNN')
    scaled_knn.clean_graph_values()

    for k in range(1, LIMIT + 1):
        scaled_knn.prediction(k, n, type='weighted')
    PlotGraph.plot_graph(scaled_knn.k_graph_values, scaled_knn.accuracy_graph_values)
    LEGEND.append('Weighted Scaled KNN')
    scaled_knn.clean_graph_values()

    basic_knn = BasicKnn('trainingData_classification.csv', 'testData_classification.csv', _plotgraph=PLOT_GRAPH)
    for k in range(1, LIMIT + 1):
        basic_knn.prediction(k)
    PlotGraph.plot_graph(basic_knn.k_graph_values, basic_knn.accuracy_graph_values)
    LEGEND.append('Basic KNN')

    weighted_knn = WeightedKnn('trainingData_classification.csv', 'testData_classification.csv', _plotgraph=PLOT_GRAPH)
    for k in range(1, LIMIT + 1):
        weighted_knn.prediction(k, n)
    PlotGraph.plot_graph(weighted_knn.k_graph_values, weighted_knn.accuracy_graph_values)
    LEGEND.append('Weighted KNN')

    PlotGraph.show_graph(LEGEND)
