from knn_model import *


class BasicKnn:

    def __init__(self,train_file, test_file, _kvalue):
        """
        :param train_file: The filename for the training instance
        :param test_file:  The filename  for  the test instance
        :param _kvalue: The K value for the KNN model

        """

        self.knn_model = Knnmodel(train_file, test_file, _kvalue)
        self.knn_model.dataset(10)
        self.k_value = _kvalue

    def prediction(self):
        """
        Calculates the euclidean distance between each query instance and the train dataset and returns accuracy
        prediction
        :return:  Accuracy of the prediction
        """

        results = np.apply_along_axis(self.knn_model.calculateDistances, 1, self.knn_model.test_data,
                                      self.knn_model.train_data)
        percentage = self.knn_model.basic_knn_percentage(results)
        print(f'The Basic KNN model with k = {self.k_value}, has and accuracy of {percentage} %')


if __name__ == '__main__':
    basic_knn = BasicKnn('trainingData_classification.csv', 'testData_classification.csv', 1)
    basic_knn.prediction()

# knnmodel = Knnmodel('trainingData_classification.csv','testData_classification.csv', 1)
#
# knnmodel.dataset(10)
#
# results = np.apply_along_axis(knnmodel.calculateDistances,1, knnmodel.test_data,knnmodel.train_data)
# percentage = knnmodel.basic_knn_percentage(results)
#
# print(f'The KNN model with k = {k_value}, has and accuracy of {percentage} %')


