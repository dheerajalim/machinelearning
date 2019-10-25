"""
Comprehensive range of techniques :
1 . Using scaling on the KNN model to see the improvement in results

"""

from knn_model import *
from part_1_oop import BasicKnn
from part_2_a import WeightedKnn


class ScaledKnn:

    def __init__(self, train_file, test_file):
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
            percentage = self.knn_model.basic_knn_percentage(self.results, k_value)
            print(f'The Scaled Basic KNN model with k = {k_value}, has and accuracy of {round(percentage, 2)} %')
            percentage = self.knn_model.weighted_knn_percentage(self.results, k_value, n_value)
            print(f'The Scaled Weighted KNN model with k = {k_value} and n = {n_value},'
                  f' has and accuracy of {round(percentage, 2)} %')
        except Exception as e:
            print(f'Error finding accuracy for K = {k_value}, error {e}')


if __name__ == '__main__':
    scaled_knn = ScaledKnn('trainingData_classification.csv', 'testData_classification.csv')
    scaled_knn.prediction(10,11)
    # basic_knn = BasicKnn('trainingData_classification.csv', 'testData_classification.csv')
    # basic_knn.prediction(1)
    # weighted_knn = WeightedKnn('trainingData_classification.csv', 'testData_classification.csv')
    # weighted_knn.prediction(1)
