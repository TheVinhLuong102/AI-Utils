import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

class StackEnsemble():
    """
    Feature-Weighted Linear Stacking by Joseph Sill, Gabor Takacs, Lester Mackey3 and David Lin
    https://arxiv.org/pdf/0911.0460.pdf

    Implemented 2 layer stack where first stack composed by arbitrary number of models (share same interface for fir/predict), 
    second stack can be arbitrary model, provided that it exposed similar fit/predict method similar to scikitlearn estimator.

    Support:
    - add aribitrary models* in the first stack
    - add arbitrary model* in the second stack
    - residual mode pass all original features in the second stack so it has more signal to ensemble

    Restrictions/Limitation:
    - all models need to exposed fit/predict methods with same signature
    - all models shared the same features set
    - if your data is too small, using StackEnsemble might not helpful. 
    - Does not provide evaluation method for each individual model.

    Examples:

    """
    def __init__(self, residual = False, features = [], target='', second_stack_model = None, train_size=0.9):
        """
        Args:
          residual: enable features to pass-through to second stack layer otherwise second stack will only consider output of first layer models.
          features: list of features name
          target: name of target of prediction
          second_stack_model: second stack model
          train_size: percentage of data will be used to train first layer models
        """
        #list of models in the first stack layer
        self.models = []
        self.second_stack_model = second_stack_model
        self.residual = residual
        self.features = features
        self.target = target
        self.train_size = train_size

    def add_model(self, model):
        """
        Args:
            model: ML model (Python object)
        """
        self.models.append(model)

    def evaluate(self, y_predict, y_true):
        #evaluation
        print("MAE : %0.05f " %mean_absolute_error(y_true, y_predict))

    def fit(self, xdata, ydata):
        """
        Args:
            xdata: features data for training, Pandas dataframe
            ydata: target data for training, Pandas dataframe
        """
        #split data for first stack and second stack
        trrows = int(self.train_size * len(xdata))
        first_stack_train_data = xdata.ix[:trrows]
        second_stack_train_data = xdata.ix[trrows:]

        first_stack_train_label = ydata.ix[:trrows]
        second_stack_train_label = ydata.ix[trrows:]

        #loop through each model and start training for first stack
        #collect prediction result
        predictions_set = []
        for model in    self.models:
            model.fit(first_stack_train_data, first_stack_train_label)
            current_prediction = model.predict(second_stack_train_data)
            #append predictions
            predictions_set.append(current_prediction)

        ytrain_data_for_second_stack = second_stack_train_label
        #prepare train data for second stack
        result = None
        xtrain_data_for_second_stack = None

        if self.residual:
            #include other features as well
            xtrain_data_for_second_stack = second_stack_train_data.copy(deep=True)
        

        for predictions in predictions_set:
            #reshape as column wide
            predictions = np.reshape(predictions, (len(predictions),1))
            if xtrain_data_for_second_stack is None:
                xtrain_data_for_second_stack = predictions
            else:
                xtrain_data_for_second_stack = np.hstack((xtrain_data_for_second_stack,predictions))



        second_target = second_stack_train_label.as_matrix()
        #evaluating each model
        for prediction in predictions_set:
            self.evaluate(prediction, second_target)

        # xtrain_data_for_second_stack.to_csv("~/data/xtrain_second_stack.csv")
        # ytrain_data_for_second_stack.to_csv("~/data/ytrain_second_stack.csv")
        self.second_stack_model.fit(xtrain_data_for_second_stack, ytrain_data_for_second_stack)

    def predict(self, data):
        """
        Args:
            data: features data for prediction, Pandas dataframe
        """
        #predict through first stack
        predictions_set = []
        for model in self.models:
            prediction = model.predict(data)
            prediction = np.reshape(prediction, (len(prediction),1))
            predictions_set.append(prediction)

        xtrain_data_for_second_stack = None
        if self.residual:
            data_to_predict = data.copy(deep=True)
            data_to_predict = data_to_predict.as_matrix()
            data_to_predict = np.reshape(data_to_predict, (len(data), len(data.columns.values)))
            xtrain_data_for_second_stack = data_to_predict
        
        #get previous layer prediction
        for prediction in predictions_set:
            if xtrain_data_for_second_stack is not None:
                xtrain_data_for_second_stack = np.hstack((xtrain_data_for_second_stack, prediction))
            else:
                xtrain_data_for_second_stack = prediction

        return self.second_stack_model.predict(xtrain_data_for_second_stack)