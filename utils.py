# -*- coding: utf-8 -*-
"""
This file includes the utilities functions

@author: mailt
"""
import numpy as np

def create_dataset(dataset, time_step = 1):
    """
    Parameters
    ----------
    dataset : array
        input dataset.
    time_step : number, optional
        The sliding window value. The default is 1.

    Returns
    -------
    array
        Input with output corresponding to input, Flattten data for SVR model.

    """
    dataX, dataY, dataX_flatten= [], [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i : (i + time_step), :]
        dataX.append(a)
        dataX_flatten.append(a.flatten())
        dataY.append(dataset[i + time_step, :])
    
    return np.array(dataX), np.array(dataY), np.array(dataX_flatten)

def train_test_split(data, train_per = 0.8):
    """
    Create Training dataset - 80%, Test - 20%, validation  - 20% of training dataset

    Parameters
    ----------
    data : Array - input splitting dataset.
    train_per : float in [0, 1], optional
        Percentage number of training dataset. The default is 0.8.

    Returns
    -------
    train_data : Array - Training dataset
    test_data : Array - Test dataset
    val_data : Array - Validation dataset

    """
    # Get size length
    
    # Get data of the last years for test
    data_length = len(data)
    test_size = 365 # fix value
    train_size = data_length - test_size
    val_size = int(train_size * 0.2)
    # 
    # train_size = int(data_length * train_per)
    # test_size = data_length - train_size
    # val_size = int(train_size * (1 - train_per))

    # Get data with size from input dataset
    train_data = data[0 : train_size, ]
    test_data = data[- data_length : , :]
    val_data = data[data_length - test_size - val_size : data_length - test_size,:]
    
    return train_data, test_data, val_data

# Waking up - return the true/false array
def classification_pred(y):
    preds = []
    for i in range(1, len(y)):
        last_y = y[i - 1]
        curr_y = y[i]
        preds.append(curr_y - last_y > 0.0 )
        
    return np.array(preds)
    
    
