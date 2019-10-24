import numpy as np
import math


train_dataset  = np.genfromtxt('trainingData_classification.csv', delimiter = ',')

# Creates all the features of the training dataset
# Removing the class column from the training dataset
train_data  = np.delete(train_dataset,10,axis = 1)

# Numpy array to store the class of training dataset
train_class = train_dataset[:,10]

test_dataset  = np.genfromtxt('testData_classification.csv', delimiter = ',')

# Creates all the features of the test dataset
# Removing the class column from the test dataset
test_data  = np.delete(test_dataset,10,axis = 1)

# Numpy array to store the class of test dataset
test_class = test_dataset[:,10]

def calculateDistances(query_instance, feature_list):
    feature_difference = feature_list - query_instance
    euclidena_distance = np.sqrt(np.sum(np.square(feature_difference),axis=1))
    sorted_distance_index = np.argsort(euclidena_distance)
    return euclidena_distance, sorted_distance_index


results = np.apply_along_axis(calculateDistances,1, test_data,train_data)

sorted_indicies = results[:,1].astype('int32')

def knn_vote(prediction):
    return np.bincount(prediction).argmax()

# Contains all the indicies representing the minimum euclidena distance
minimum_dist = sorted_indicies[:,:k]

# Numpy array to store the classes predicted for the test data
prediction = train_class[minimum_dist].astype('int32')

# Finding the mode of the classes in K neighbours
find_res = np.apply_along_axis(knn_vote,1, prediction)

# Calculating the count of correct predictions
correct_prediction = np.count_nonzero(test_class == find_res)

# The percentage of correct prediction
percentage =( correct_prediction/len(test_dataset) ) *100

print(f'The model has an accuracy of {percentage} %')