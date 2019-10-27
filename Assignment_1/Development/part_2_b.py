"""
Comprehensive range of techniques :
1 . Using scaling on the KNN model to see the improvement in results
    Scaling on Basic KNN
    Scaling on Weighted KNN
2. Different Distance Metrics
3. Different K parameters

"""

from knn_model import *


class ScaledKnn:

    def __init__(self, train_file, test_file, _scaling=False, _distance='E', _plotgraph=False):
        """
        :param train_file: The filename for the training instance
        :param test_file:  The filename  for  the test instance
        :param _kvalue: The K value for the KNN model

        """
        self.knn_model = Knnmodel(train_file, test_file)
        self.knn_model.dataset(10)
        self.scaling = _scaling
        self.distance = _distance

        if _scaling:
            self.knn_model.dataset_scaling()
            if _distance == 'E':
                self.results = np.apply_along_axis(self.knn_model.calculateDistances, 1,
                                                   self.knn_model.scaled_test_data,
                                                   self.knn_model.scaled_train_data)
            elif _distance == 'M':
                self.results = np.apply_along_axis(self.knn_model.manhattanDistance, 1, self.knn_model.scaled_test_data,
                                                   self.knn_model.scaled_train_data)
        else:
            if _distance == 'E':
                self.results = np.apply_along_axis(self.knn_model.calculateDistances, 1, self.knn_model.test_data,
                                                   self.knn_model.train_data)
            elif _distance == 'M':
                self.results = np.apply_along_axis(self.knn_model.manhattanDistance, 1, self.knn_model.test_data,
                                                   self.knn_model.train_data)

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
            distance_type = 'Euclidean' if self.distance == 'E' else 'Manhattan'
            if type == 'basic':
                percentage = self.knn_model.basic_knn_percentage(self.results, k_value)
                if self.scaling:
                    print(f'The Scaled Basic KNN model with k = {k_value} with {distance_type},'
                          f' has an accuracy of {round(percentage, 2)} %')
                else:
                    print(f'The Basic KNN model with k = {k_value} with {distance_type},'
                          f' has an accuracy of {round(percentage, 2)} %')

            elif type == 'weighted':
                percentage = self.knn_model.weighted_knn_percentage(self.results, k_value, n_value)
                if self.scaling:
                    print(f'The Scaled Weighted KNN model with k = {k_value} and n = {n_value} with {distance_type},'
                          f' has an accuracy of {round(percentage, 2)} %')
                else:
                    print(f'The Basic Weighted KNN model with k = {k_value} and n = {n_value} with {distance_type},'
                          f' has an accuracy of {round(percentage, 2)} %')

            if self.plot_graph:
                self.accuracy_graph_values.append(round(percentage, 2))
                self.k_graph_values.append(k_value)

        except Exception as e:
            print(f'Error finding accuracy for K = {k_value}, error {e}')

    def clean_graph_values(self):
        self.accuracy_graph_values = []
        self.k_graph_values = []


def run_model(model, limit, type, n_value=1):
    for k in range(1, limit + 1):
        scaled_knn.prediction(k, n_value, type=type)
    PlotGraph.plot_graph(model.k_graph_values, model.accuracy_graph_values)
    scaled_knn.clean_graph_values()


if __name__ == '__main__':
    PLOT_GRAPH = True

    LEGEND = []
    LIMIT = 60
    n = 2

    scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _scaling=False,
                           _distance='E', _plotgraph=PLOT_GRAPH)
    LEGEND.append('Unscaled Basic KNN Euclidean')
    run_model(scaled_knn, LIMIT, 'basic')

    scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _scaling=True,
                           _distance='E', _plotgraph=PLOT_GRAPH)
    LEGEND.append('Scaled Basic KNN Euclidean')
    run_model(scaled_knn, LIMIT, 'basic')
    PlotGraph.show_graph(LEGEND, filename='Scaled_basic')

    LEGEND = []
    scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _scaling=False,
                           _distance='E', _plotgraph=PLOT_GRAPH)
    LEGEND.append('Unscaled Weighted KNN Euclidean')
    run_model(scaled_knn, LIMIT, 'weighted', n)

    scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _scaling=True,
                           _distance='E', _plotgraph=PLOT_GRAPH)
    LEGEND.append('Scaled Weighted KNN Euclidean')
    run_model(scaled_knn, LIMIT, 'weighted', n)
    PlotGraph.show_graph(LEGEND, filename='Scaled_weighted', n_value=n)

    """
    ######==========Uncomment the below code to generate the graphs in section 3.1 of report.=====######
    """

    # """
    # I ) Applying performance techniques on:
    # 1. Unscaled Basic KNN with K values Euclidean Distance
    # 2. Unscaled Basic KNN with K values Manhattan Distance
    # 3. Scaled Basic KNN with K values Euclidean Distance
    # 4. Scaled Basic KNN with K values Manhattan Distance
    #
    # """
    #
    # scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _scaling=False,
    #                        _distance='E', _plotgraph=PLOT_GRAPH)
    # LEGEND.append('Unscaled Basic KNN Euclidean')
    # run_model(scaled_knn, LIMIT, 'basic')
    #
    # scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _scaling=False,
    #                        _distance='M', _plotgraph=PLOT_GRAPH)
    # LEGEND.append('Unscaled Basic KNN Manhattan')
    # run_model(scaled_knn, LIMIT, 'basic')
    #
    # scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _scaling=True,
    #                        _distance='E', _plotgraph=PLOT_GRAPH)
    # LEGEND.append('Scaled Basic KNN Euclidean')
    # run_model(scaled_knn, LIMIT, 'basic')
    #
    # scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _scaling=True,
    #                        _distance='M', _plotgraph=PLOT_GRAPH)
    # LEGEND.append('Scaled Basic KNN Manhattan')
    # run_model(scaled_knn, LIMIT, 'basic')
    #
    # PlotGraph.show_graph(LEGEND, filename='All_basic')
    #
    # """
    # II ) Applying performance techniques on:
    # 1. Unscaled Weighted KNN with K values Euclidean Distance
    # 2. Unscaled Weighted KNN with K values Manhattan Distance
    # 3. Scaled Weighted KNN with K values Euclidean Distance
    # 4. Scaled Weighted KNN with K values Manhattan Distance
    #
    # """
    # LEGEND = []
    # scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _scaling=False,
    #                        _distance='E', _plotgraph=PLOT_GRAPH)
    # LEGEND.append('Unscaled Weighted KNN Euclidean')
    # run_model(scaled_knn, LIMIT, 'weighted', n)
    #
    # scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _scaling=False,
    #                        _distance='M', _plotgraph=PLOT_GRAPH)
    # LEGEND.append('Unscaled Weighted KNN Manhattan')
    # run_model(scaled_knn, LIMIT, 'weighted', n)
    #
    # scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _scaling=True,
    #                        _distance='E', _plotgraph=PLOT_GRAPH)
    # LEGEND.append('Scaled weighted KNN Euclidean')
    # run_model(scaled_knn, LIMIT, 'weighted', n)
    #
    # scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv', _scaling=True,
    #                        _distance='M', _plotgraph=PLOT_GRAPH)
    # LEGEND.append('Scaled weighted KNN Manhattan')
    # run_model(scaled_knn, LIMIT, 'weighted', n)
    #
    # PlotGraph.show_graph(LEGEND, filename='All_weighted',n_value=n)

