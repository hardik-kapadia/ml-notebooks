import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loss_functions import *

def gradient_descent(actual_values: np.array, training_data: pd.DataFrame, loss_function=mean_squared_error, learning_rate: float = 1, learning_limiter: float = 1, minimum_loss_difference=0.00005) -> tuple:

    if(len(actual_values) != len(training_data.index)):
        raise ValueError(
            'no. of actual values should be the same as number of feature vectors')

    def partial_derivative(wrt: int) -> float:

        if(len(actual_values) != len(training_data.index)):
            raise ValueError(
                'no. of actual values should be the same as number of feature vectors')

        if(training_data.shape[1] != len(coefs)):
            raise ValueError(
                'Number of coefs should match number of independent variables')

        # print(f'coefs are: {coefs}')

        d = 0
        for_constant = wrt == -1

        def multiplier(
            arr: np.array) -> float: return 1 if for_constant else float(arr[wrt])

        for index, row in training_data.iterrows():

            r = row.to_numpy()
            t = actual_values[index]

            for i in range(r.size):
                t = (t - (coefs[i]*r[i]))

            t = t*multiplier(r)

            d += t

        n = len(actual_values)
        d = d * -2/n

        return d

    cols = training_data.shape[1]
    coefs = [0 for i in range(cols)]
    constant = 0

    count = 0

    loss_vals = []

    prev = 0
    loss = 0

    while True:

        # print(f'at count {count} -> {coefs} and {constant}')

        for i in range(len(coefs)):

            p_d = partial_derivative(i)

            coefs[i] = coefs[i] - (learning_rate * p_d)

        c_pd = partial_derivative(-1)

        constant = constant - (learning_rate * c_pd)

        loss = loss_function(actual_values, training_data, coefs, constant)

        loss_vals.append(loss)

        if loss <= learning_limiter:
            break

        if count > 0 and abs(loss - prev) < minimum_loss_difference:
            break

        count += 1

        prev = loss
    

    return (coefs, constant, loss_vals)


class Linear_Regression:

    def __init__(self, actual_values: np.array = np.array([]), training_data: pd.DataFrame = pd.DataFrame.from_dict({}), loss_function=mean_squared_error, learning_rate: float = 1, learning_limiter: float = 1):
        self.dependent_variable = actual_values
        self.training_data = training_data
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.learning_limiter = learning_limiter

    def fit(self, actual_values: np.array, training_data: pd.DataFrame):
        self.dependent_variable = actual_values
        self.training_data = training_data

    def train(self):
        self.coefs, self.intercept, self.loss_vals = gradient_descent(
            actual_values=self.dependent_variable, training_data=self.training_data, learning_rate=self.learning_rate, learning_limiter=self.learning_limiter)

    def loss(self):
        return self.loss_function(self.dependent_variable, self.training_data, self.coefs, self.intercept)

    def rsme(self):
        return np.sqrt(self.loss())

    def predict(self, current_feature_vector: np.array):
        return np.sum(np.multiply(current_feature_vector, self.coefs)) + self.intercept
    
    def plot_gradient_descent(self):
        plt.plot(self.loss_vals)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Gradient descent')
        plt.show()
