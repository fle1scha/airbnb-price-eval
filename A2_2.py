import argparse
import time

from keras.optimizers import adam_v2
from art import tprint
from sklearn.model_selection import KFold
from sklearn.metrics import *
from keras.backend import argmax
from numpy.lib.arraysetops import unique
import pandas as pd
import numpy as np
from NeuralNets import *
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler


def loadData(file):
    df = pd.read_csv(file)
    # print(df.head(5))
    df = df.drop(labels='Unnamed: 0', axis=1)
    # print(df.head(5))
    return df


def selectData(df):

    df = df[[
        'superhost', 'ward', 'pax', 'bedrooms', 'beds', 'bathrooms', 'price',
        'review_rating', 'latitude', 'longitude', 'entire_home', 'hotel_room',
        'shared_room', 'mininmum_nights', 'maximum_nights'
    ]]

    df = df.dropna()
    # print(df.isnull().values.any())
    return df


def cleanData(df):

    # Work Rate
    df['superhost'] = pd.Categorical(df['superhost']).astype('float64')
    # print(df.isnull().values.any())

    # Check types and return
    print(df.dtypes)

    return df


def EDA(df):

    # percentile list
    perc = [.20, .40, .60, .80]

    # list of dtypes to include
    include = ['object', 'float', 'int']

    # calling describe method
    desc = df.describe(percentiles=perc, include=include)

    # display
    # print(desc)

    # df.boxplot(column=['bedrooms', 'bathrooms'])
    # plt.show()

    # sns.kdeplot(df['price'])
    # plt.title('Kernel Density Plot of Price')
    # plt.show()

    # sns.kdeplot(df['review_rating'])
    # plt.title('Kernel Density Plot of Review Rating')
    # plt.show()

    # Remove incorrectly recorded and outlying values

    # df = df[df.price <= 37500]
    df = df[df.beds < 40]
    df = df[df.bathrooms < 20]

    # sns.kdeplot(df['longitude'],
    #         df['latitude'],
    #         color='r', shade=True,
    #         cmap="Reds", shade_lowest=False)
    # plt.show()

    # df.plot(x='longitude',
    #         y='latitude',
    #         color=df['price'].values,
    #         cmap="jet",
    #         title='Prices of Airbnbs in Cape Town',
    #         kind='scatter')
    # plt.show()

    # df.plot(x='longitude',
    #         y='latitude',
    #         color=df['bedrooms'].values,
    #         cmap="jet",
    #         title='Sizes of Airbnbs in Cape Town',
    #         kind='scatter')
    # plt.show()

    # df.plot(x='ward', y='price', kind = 'scatter')
    # plt.show()

    cut_bins = [0, 500 , 1250, 2000, 2750, 3500, 4250, 5000, 5750, 15000, 1000000]

    bin_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    df['price'] = pd.cut(df['price'], bins = cut_bins, labels=bin_labels)


    return df


def prepareData(df):
    # Remove Overall

    X = df.drop(columns=['price']).copy()
    Y = df['price'].values

    # print(np.unique(Y))
    

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    encoder = LabelEncoder()
    
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_train = np_utils.to_categorical(encoded_Y)

    encoder.fit(y_test)
    encoded_Y = encoder.transform(y_test)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_test = np_utils.to_categorical(encoded_Y)

    x_train = x_train.reshape(9001, 14).astype("float32")
    x_test = x_test.reshape(2251, 14).astype("float32")

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    print("The dimensions of the datasets are:")
    print("Training X:", x_train.shape, "Training Y:", dummy_y_train.shape, end="\n")
    print("Testing X:", x_test.shape, "Testing Y:", dummy_y_test.shape, end="\n")
                                              

    return x_train, y_train, dummy_y_train, x_test, y_test, dummy_y_test


def main():
    tprint("Khayamali")
    print("AirBnb Pricing and Valuation Tool")
    print("© Fleischer-Gemini 2021")
    print("-" * 40, end="\n")

    df = loadData('/Users/jessiebosman/Desktop/AI/A2/airbnbCapeTown.csv')
    df = selectData(df)
    df = cleanData(df)
    df = EDA(df)

    x_train, y_train, dummy_y_train, x_test, y_test, dummy_y_test = prepareData(df)
    # print(X), print(y)
    

    # Naive model:
    naive_model = naiveModel(df, x_train, dummy_y_train)
    naive_predictions = pd.Series(naive_model.predict(verbose=True))


    # Shallow fully-connected FFNN:
    baseline_model = baselineModel()
    baseline_model.addLayers(5, 2)
    baseline_model.train(loss_function='categorical_crossentropy',
                         optimizer='adam',
                         epochs=10,
                         batch_size=32,
                         X=x_train,
                         y=dummy_y_train,
                         verbose=True)
    baseline_predictions = [(argmax(probabilities).numpy() + 1)
                            for probabilities in baseline_model.predict(x_train)]

    # Generate generalization metrics
    score = baseline_model.evaluate(x_test, dummy_y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


    # Deeper fully-connected FFNN:
    full_model = baselineModel()
    full_model.addLayers(10, 4)
    full_model.train(loss_function='categorical_crossentropy',
                     optimizer='adam',
                     epochs=100,
                     batch_size=64,
                     X=x_train,
                     y=dummy_y_train,
                     verbose=True)
    full_predictions = [(argmax(probabilities).numpy() + 1)
                        for probabilities in full_model.predict(x_train)]

    # Generate generalization metrics
    score = full_model.evaluate(x_test, dummy_y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    # Combine predictions:
    naive_metrics = {
        'Accuracy': accuracy_score(y_train, naive_predictions),
        'Precision': precision_score(y_train, naive_predictions, average='macro'),
        'Recall': recall_score(y_train, naive_predictions, average='macro')
    }
    print(naive_metrics)

    baseline_metrics = {
        'Accuracy': accuracy_score(y_train, baseline_predictions),
        'Precision': precision_score(y_train, baseline_predictions, average='macro'),
        'Recall': recall_score(y_train, baseline_predictions, average='macro')
    }
    print(baseline_metrics)

    fullmodel_metrics = {
        'Accuracy': accuracy_score(y_train, full_predictions),
        'Precision': precision_score(y_train, full_predictions, average='macro'),
        'Recall': recall_score(y_train, full_predictions, average='macro')
    }
    print(fullmodel_metrics)

    frame = {'Naive': naive_predictions, 'Baseline NN': baseline_predictions}
    pred_df = pd.DataFrame(frame)
    # print(pred_df)

    # df.plot(x='longitude',
    #         y='latitude',
    #         color=baseline_predictions,
    #         cmap="jet",
    #         title='Price Predictions of Airbnbs in Cape Town',
    #         kind='scatter',
    #         label='price')

    # plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="EUR/USD Trading Decision Support Tool")
    # parser.add_argument('--naive', action='store_true', help="Bayesian network inference")
    # parser.add_argument('--baseline', action='store_true', help="Decision network decision support")
    # args = parser.parse_args()
    main()