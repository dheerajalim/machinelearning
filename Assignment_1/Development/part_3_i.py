from knn_model import *
import time


class WeightedRegressionKnn:

    def __init__(self, train_file, test_file):
        """
        :param train_file: The filename for the training instance
        :param test_file:  The filename  for  the test instance
        :param _kvalue: The K value for the KNN model

        """

        self.knn_model = Knnmodel(train_file, test_file)
        self.knn_model.dataset(12)

        self.results = np.apply_along_axis(self.knn_model.calculateDistances, 1, self.knn_model.test_data,
                                           self.knn_model.train_data)

    def prediction(self, k_value=1, n_value=1):
        """
        Calculates the euclidean distance between each query instance and the train dataset and returns accuracy
        prediction
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
                  f'has and accuracy of {round(percentage, 2)} %')
        except Exception as e:
            print(f'Error finding accuracy for K = {k_value}, error {e}')


if __name__ == '__main__':
    weighted_regression_knn = WeightedRegressionKnn('trainingData_regression.csv', 'testData_regression.csv')
    weighted_regression_knn.prediction(3, 3)
