from knn_model import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class FeatureSelectionKnn:

    def __init__(self,train_file, test_file):
        """
        :param train_file: The filename for the training instance
        :param test_file:  The filename  for  the test instance
        :param _kvalue: The K value for the KNN model

        """
        self.knn_model = Knnmodel(train_file, test_file)
        self.knn_model.dataset(10)
        self.train_data_updated = np.empty
        self.test_data_updated = np.empty
        self.results = None

    def feature_selection(self,number_features=5):
        selector = SelectKBest(chi2, k=number_features)
        self.train_data_updated = selector.fit_transform(self.knn_model.train_data, self.knn_model.train_class)
        feature_names = [i for i in range(0,len(self.knn_model.test_data[0]))]

        mask = selector.get_support()  # list of booleans
        new_features = []  # The list of your K best features
        for bool, feature in zip(mask, feature_names):
            if bool:
                new_features.append(feature)

        self.test_data_updated = self.knn_model.test_data[:, new_features]
        self.distance_calculation()
        print(f'Number of features selected : {len(new_features)}')
        print(f'Selected features [Columns]: {new_features}')

    def distance_calculation(self):
        self.results = np.apply_along_axis(self.knn_model.calculateDistances, 1, self.test_data_updated,
                                           self.train_data_updated)


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
    basic_knn = FeatureSelectionKnn('trainingData_classification.csv', 'testData_classification.csv')
    basic_knn.feature_selection(8)
    # for k in range(1,12):
    basic_knn.prediction(2)




