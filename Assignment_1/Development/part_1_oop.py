from knn_model import *


class BasicKnn:

    def __init__(self,train_file, test_file):
        """
        :param train_file: The filename for the training instance
        :param test_file:  The filename  for  the test instance
        :param _kvalue: The K value for the KNN model

        """

        self.knn_model = Knnmodel(train_file, test_file)
        self.knn_model.dataset(10)
        self.results = np.apply_along_axis(self.knn_model.calculateDistances, 1, self.knn_model.test_data,
                                      self.knn_model.train_data)

    def prediction(self, k_value=1):
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
            print(f'The Basic KNN model with k = {k_value}, has and accuracy of {round(percentage,2)} %')
        except Exception as e:
            print(f'Error finding accuracy for K = {k_value}, error {e}')


if __name__ == '__main__':
    basic_knn = BasicKnn('trainingData_classification.csv', 'testData_classification.csv')
    for k in range(1,12):
        basic_knn.prediction(k)

# knnmodel = Knnmodel('trainingData_classification.csv','testData_classification.csv', 1)
#
# knnmodel.dataset(10)
#
# results = np.apply_along_axis(knnmodel.calculateDistances,1, knnmodel.test_data,knnmodel.train_data)
# percentage = knnmodel.basic_knn_percentage(results)
#
# print(f'The KNN model with k = {k_value}, has and accuracy of {percentage} %')


