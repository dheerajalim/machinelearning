"""
List of Parameters for the KNN Model
"""


class Parameters:

    # Dataset Location parameters for Classification
    TRAIN_DATA_CLASSIFICATION = "data//classification//trainingData.csv"
    TEST_DATA_CLASSIFICATION = "data//classification//testData.csv"

    # Dataset Location parameters for Regression
    TRAIN_DATA_REGRESSION = "data//regression//trainingData.csv"
    TEST_DATA_REGRESSION = "data//regression//testData.csv"

    # Execution parameters
    PLOT_GRAPH = True
    LEGEND = []
    LIMIT = 60
    n = 2

    # Feature Parameter for Regression Model
    FEATURES = 10

