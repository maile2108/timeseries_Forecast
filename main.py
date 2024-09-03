# -*- coding: utf-8 -*-

# Declare library
import numpy as np
import pandas as pd # data processing lib
import matplotlib.pyplot as plt # visualization lib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

from models import LSTM_Model, CNNLSTM, RNN
from visualisation import Visualisation
from utils import *


# Global variables
time_step = 50
num_feartures = 3
output_dim = 3 # multivariate output

# Declare Models
CNNLSTM = CNNLSTM(time_step, num_feartures,  output_dim)
CNNLSTM_model = CNNLSTM.model

LSTM = LSTM_Model(time_step, num_feartures,  output_dim)
LSTM_model = LSTM.model

RNN = RNN(time_step, num_feartures,  output_dim)
RNN_model = RNN.model

# Load data from dataset file
data_frame = pd.read_csv('dataset.csv', index_col = 'Date', parse_dates = True)
data_frame.info()


##############################################################################
##############################################################################
# -*- Data visualisation -*-
# Including: original data display, data visualisation by year, month
# And the data distribution
data_visual = Visualisation(data_frame)

# You can uncomment to see the plots as well as the saving pdf files
data_visual.original_plot()
# data_visual.year_plot()
# data_visual.distribution_plot()
# data_visual.data_boxplot()


#############################################################################
#############################################################################
# -*- Prediction Section -*-

# Scalar for Normalization in range (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
# -*-Process with original value-*-
#dataFrame_Org = data_frame.fillna(0) # Fill NaN value by 0
dataFrame_Org = data_frame.dropna() # Drop NaN value

# Output
y = dataFrame_Org[['Consumption','Wind', 'Solar']]
y = scaler.fit_transform(y) # Normalization

# Training, test, validate dataset
# Split dataset
train_data, test_data, val_data = train_test_split(y) 


# Create(X - input, Y-output) 
X_train, Y_train, X_train_flatten = create_dataset(train_data, time_step)
X_test, Y_test,_ = create_dataset(test_data, time_step)
X_val, Y_val,_ = create_dataset(val_data, time_step)

# Model training for Consumption
CNNLSTM_history = CNNLSTM_model.fit(X_train, Y_train, validation_data = (X_val, Y_val),
                                    epochs = 100)

LSTM_history = LSTM_model.fit(X_train, Y_train,
                                    validation_data = (X_val, Y_val),
                                    epochs = 100)

RNN_history = RNN_model.fit(X_train, Y_train, validation_data = (X_val, Y_val),
                                    epochs = 100)

# Plot Loss and validation loss of proposed models
plt.plot(CNNLSTM_history.history['loss'], 'b-', label='Train') # tb
plt.plot(CNNLSTM_history.history['val_loss'], 'r--', label='Validation') # tb

plt.legend(loc = 1,  ncol = 1)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(color='gray', linestyle='-.', linewidth = 0.2)
plt.savefig('./Loss_multi.pdf', bbox_inches='tight') # Save pdf file
plt.show()

#  Predict and check performance metrics

# Get mean error for tesing set 
#CNN-LSTM
Y_train_cnn  = CNNLSTM_model.predict(X_train)
Y_test_cnn = CNNLSTM_model.predict(X_test)

# LSTM
Y_train_lstm  = LSTM_model.predict(X_train)
Y_test_lstm = LSTM_model.predict(X_test)

# RNN
Y_test_rnn= RNN_model.predict(X_test)
Y_train_rnn  = RNN_model.predict(X_train)

# Error for testing data
rmse_train_cnn = mean_squared_error(Y_train, Y_train_cnn, squared = True)
rmse_train_lstm = mean_squared_error(Y_train, Y_train_lstm, squared=True)
rmse_train_rnn = mean_squared_error(Y_train, Y_train_rnn, squared=True)

rmse_test_cnn = mean_squared_error(Y_test, Y_test_cnn, squared = True)
rmse_test_lstm = mean_squared_error(Y_test, Y_test_lstm, squared = True)
rmse_test_rnn = mean_squared_error(Y_test, Y_test_rnn, squared = True)

mape_cnn = mean_absolute_percentage_error(Y_test, Y_test_cnn)
mape_lstm = mean_absolute_percentage_error(Y_test, Y_test_lstm)
mape_rnn = mean_absolute_percentage_error(Y_test, Y_test_rnn)


print('Consumption Test error by CNN:', rmse_test_cnn)
print('Consumption Test error by LSTM:', rmse_test_lstm)
print('Consumption Test error by RNN:', rmse_test_rnn)

# Plot BAR-type Loss of training, testing and validation of proposed models
model_name = ['CNN-LSTM', 'LSTM', 'RNN']
model_value =  [rmse_test_cnn, rmse_test_lstm, rmse_test_rnn]
bar_colors = ['tab:orange', 'tab:blue', 'tab:red']
plt.bar(model_name, model_value, color=bar_colors)
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('MSE')
plt.ylim(0, 0.025)
plt.grid(color='gray', linestyle='-.', linewidth = 0.2)
plt.savefig('./errBar.pdf', bbox_inches='tight') # Save pdf file
plt.show()


# Display train, test and validate prediction
# Transform back to original form
# For Consumption

Y_train_predict = scaler.inverse_transform(Y_train_cnn)
Y_train_predict_lstm = scaler.inverse_transform(Y_train_lstm)
Y_train_predict_rnn = scaler.inverse_transform(Y_train_rnn)

Y_test_predict = scaler.inverse_transform(Y_test_cnn)
Y_test_predict_lstm = scaler.inverse_transform(Y_test_lstm)
Y_test_predict_rnn  = scaler.inverse_transform(Y_test_rnn)
# Y_val_predict = scaler.inverse_transform(Y_val_cnn)

Y_train_actual = scaler.inverse_transform(Y_train)
# Y_test_actual = scaler.inverse_transform(Y_test)
# Y_val_actual = scaler.inverse_transform(Y_val)

# Get the 366 last data to draw
dayofyear = 90
Y_cnn_display = Y_test_predict[- dayofyear :]
Y_cnn_display = Y_cnn_display.T # Transpose data to get value of each column

Y_lstm_display = Y_test_predict_lstm[- dayofyear :]
Y_lstm_display = Y_lstm_display.T

Y_rnn_display = Y_test_predict_rnn[- dayofyear :]
Y_rnn_display = Y_rnn_display.T

Y_act_display = Y_train_actual[- dayofyear :]
Y_act_display = Y_act_display.T

# Plot actual value and predic value in the last whole year
label_actual = ['Consumption - actual','Wind - actual', 'Solar - actual']
label_CNN = ['Consumption - CNNLSTM','Wind - CNNLSTM', 'Solar - CNNLSTM']
label_LSTM = ['Consumption - LSTM','Wind - LSTM', 'Solar - LSTM']
label_RNN = ['Consumption - RNN','Wind - RNN', 'Solar - RNN']

# Plot prediction and actual data by 3 proposed models
for i in range(len(Y_cnn_display)):
    plt.plot(Y_cnn_display[i],'-', label = label_CNN[i]) # CNN LSTM
    plt.plot(Y_lstm_display[i], '--', label = label_LSTM[i]) # LSTM
    plt.plot(Y_rnn_display[i], '-.', label = label_RNN[i]) # RNN
    plt.plot(Y_act_display[i], ':', label = label_actual[i])

plt.xlabel('Time Step')
plt.ylabel('Consumption/Production')
plt.grid(color='gray', linestyle='-.', linewidth = 0.2)
plt.legend(loc = 2,  ncol = 3, prop = {'size': 5}, bbox_to_anchor = (0.0,0.58))
plt.savefig('./AcPre.pdf', bbox_inches='tight') # Save pdf file
plt.show()
    
#############################################################################
#############################################################################
#  -*- Future Consumption/Production forecasting -*- 
temp_input = test_data
# Predict for next 30 days
output = []
days = 366
i = 0
while(i < days):
    x_previous = temp_input[- time_step :] # get time span equals to time_step
    # reshape as [[[sample, time-step, features]]]
    x_previous = x_previous.reshape((1, x_previous.shape[0],  x_previous.shape[1])) 
    
    y_out = CNNLSTM_model.predict(x_previous)
    
    # Push predicted value to template data
    temp_input = np.vstack([temp_input, y_out[0].tolist()])
    
    output.append(y_out[0].tolist())
    i = i + 1

# Plot test data and prediction data of whole current year

day_old = range(days) # X range for old data
day_new = range(days, days + days) # X range for predicted data

Y_test_display = Y_test_predict[- days : ]
Y_test_display = Y_test_display.T
output = scaler.inverse_transform(output)
output = output.T # Transpose data to get value as columns

# For total consumption
plt.plot(day_old, Y_test_display[0], 'blue', label = 'Consumption- old data')
plt.plot(day_new, output[0], 'orange', label='Consumption - prediction')

# For Wind production
plt.plot(day_old, Y_test_display[1], 'green', label='Wind - old data')
plt.plot(day_new, output[1], 'm', label = 'Wind - prediction')

# For Solar production
plt.plot(day_old, Y_test_display[2], 'black', label='Solar - old data')
plt.plot(day_new, output[2], 'red', label='Solar - prediction')

#plt.plot(x_train, Y_train_actual.flatten(), 'black', x_test, Y_test_actual.flatten(), 'green')
plt.xlabel('Time Step')
plt.ylabel('Average Consumption')
plt.grid(color = 'gray', linestyle = '-.', linewidth = 0.2)
plt.legend(loc = 2, ncol = 3, prop = {'size': 6}, bbox_to_anchor = (0.0,0.58))
# suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
plt.savefig('./Predic.pdf', bbox_inches = 'tight')
plt.show()


