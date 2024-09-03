# -*- coding: utf-8 -*-
"""
Build models including Long-Short Term 
@author: MaiLe
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Conv1D, MaxPooling1D


#  Create LSTM model
class LSTM_Model():
    def __init__(self, input_dim, num_feartures, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_feartures = num_feartures
        self.model = self.model()
    
    def model(self):
        
        model = Sequential()
        
        # The first LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 128, input_shape = (self.input_dim, self.num_feartures)))
        
        # Adding the output layer
        model.add(Dense(units = self.output_dim))
        
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        return model


# Create CNN-LSTM model
class CNNLSTM():
    def __init__(self, input_dim, num_feartures, output_dim, scaler = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_feartures = num_feartures
        self.model = self.model()
    
    def model(self):
        
        model = Sequential()
        
        # The first Conv
        model.add(Conv1D(64, 2,
                       strides=1,
                       padding='same',
                       input_shape=(self.input_dim, self.num_feartures)))
        
        # The second Conv, 64 - nodes, 2 is size of the convolution window
        # When padding="same" and strides=1, the output has the same size as the input.
        model.add(Conv1D(64, 2,
                   strides = 1,
                   padding = 'same',
                   input_shape=(self.input_dim, self.num_feartures)))
        
        # The pooling layer
        # "valid" means no padding
        model.add(MaxPooling1D(pool_size = 2, padding = 'valid'))
        
        # LSTM model
        model.add(LSTM(128,  input_shape=(self.input_dim, self.num_feartures)))
        model.add(Dense(self.output_dim)) # Output layer
        
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        
        return model

# Simple vanilla RNN
class RNN():
    def __init__(self, input_dim, num_feartures, output_dim, scaler = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_feartures = num_feartures
        self.model = self.model()
    
    def model(self):
        
        model = Sequential()

        model.add(SimpleRNN(128)) # 64 is the hidden node
    
        model.add(Dense(self.output_dim))
        
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        
        return model
     

if __name__ == '__main__':
    model = LSTM_Model(100, 5)
    