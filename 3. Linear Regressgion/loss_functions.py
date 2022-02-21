import pandas as pd
import numpy as np

def mean_squared_error(actual_values: np.array, training_data: pd.DataFrame, coefs: list, constant: int) -> float:

    if(len(actual_values) != len(training_data.index)):
        raise ValueError(
            'no. of actual values should be the same as number of feature vectors')

    if(training_data.shape[1] != len(coefs)):
        raise ValueError(
            'Number of coefs should match number of independent variables')

    loss = 0
    n = len(actual_values)

    for index, row in training_data.iterrows():

        r = row.to_numpy()
        t = actual_values[index]

        for i in range(r.size):
            t = (t - (coefs[i] * r[i]))

        t -= constant
        loss += t*t

    loss /= n

    return loss

def absolute_loss(actual_values: np.array, training_data: pd.DataFrame, coefs: list, constant: int):
    if(len(actual_values) != len(training_data.index)):
        raise ValueError(
            'no. of actual values should be the same as number of feature vectors')

    if(training_data.shape[1] != len(coefs)):
        raise ValueError(
            'Number of coefs should match number of independent variables')
        
    loss = 0
    n = len(actual_values)

    for index, row in training_data.iterrows():

        r = row.to_numpy()
        t = actual_values[index]

        for i in range(r.size):
            t = (t - (coefs[i] * r[i]))

        t -= constant
        loss += abs(t)

    loss /= n

    return loss