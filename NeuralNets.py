from keras.backend import argmax
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from keras.layers import BatchNormalization




class baselineModel:

    def __init__(self):
        self.model = Sequential()

    def addLayers(self, nodes, layers):

        print("Configuring baseline neural network...")
        # Input layer with 14 feature nodes.
        self.model.add(BatchNormalization())
        self.model.add(Dense(nodes, input_dim=14, activation='relu'))

        # Add layers number of Dense, fully connected layers.
        for i in range(layers):
            self.model.add(BatchNormalization())
            self.model.add(Dense(nodes, activation = 'relu'))

        # Output layer with 10 nodes.
        self.model.add(Dense(10, activation='softmax'))


    def train(self, loss_function, optimizer, epochs, batch_size, X, y, verbose):
        self.model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose = verbose)
        
    def evaluate(self, X, y, verbose):
        score = self.model.evaluate(X, y, verbose=verbose)
        return score

    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions
        
    
class naiveModel:
    
    def __init__(self, df, X, y):
        self.df = df
        self.X = X
        self.y = y
    
    def predict(self, verbose):

        price_frequency = self.df['price'].value_counts()
        price_proportions = self.df['price'].value_counts(normalize=True)

        if verbose:
            print('Initiating naive model...')
            print("The value counts are:", price_frequency, "", sep = '\n' )
            print("The normalized proportions are:", price_proportions, "", sep = '\n' )

        naive_predictions = [2 for i in range(len(self.X))] 

        return naive_predictions


        