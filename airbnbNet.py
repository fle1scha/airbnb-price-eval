import argparse
from keras.layers import BatchNormalization
from keras.layers.core import Dropout
from statistics import mean
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import *
from keras.backend import argmax
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import Sequential

class naiveModel:
    
    def __init__(self, df, X, y):
        self.df = df
        self.X = X
        self.y = y
    
    def predict(self, verbose, X):

        price_frequency = self.df['price'].value_counts()
        price_proportions = self.df['price'].value_counts(normalize=True)

        if verbose:
            print('Initiating naive model...')
            print("The value counts are:", price_frequency, "", sep = '\n' )
            print("The normalized proportions are:", price_proportions, "", sep = '\n' )

        naive_predictions = [1 for i in range(len(X))] 

        return naive_predictions

def loadData(file):
    df = pd.read_csv(file)
    df = df.drop(labels='Unnamed: 0', axis=1)
    return df

def selectData(df):

    df = df[[
        'superhost', 'pax', 'bedrooms', 'price', 'review_rating', 'latitude',
        'longitude', 'entire_home', 'hotel_room', 'shared_room',
        'mininmum_nights', 'maximum_nights', 'availability_365',
        'number_of_reviews', 'fullyBooked', 'instant_bookable'
    ]]

    df = df.dropna()
    return df

def cleanData(df):

    df['superhost'] = pd.Categorical(df['superhost']).astype('float64')
    df['fullyBooked'] = pd.Categorical(df['fullyBooked']).astype('float64')
    df['instant_bookable'] = pd.Categorical(
        df['instant_bookable']).astype('float64')

    return df

def EDA(df):

    view = input("View EDA graphs [Y/N]: ")
    if view == 'Y':
        # percentile list
        perc = [.20, .40, .60, .80]

        # list of dtypes to include
        include = ['object', 'float', 'int']

        # calling describe method
        desc = df.describe(percentiles=perc, include=include)

        print(desc)

        # bedrooms and pax boxplots
        df.boxplot(column=['bedrooms', 'pax'])
        plt.title("Boxplots of Bedrooms & Pax")
        plt.show()

        df.boxplot(column=['price'])
        plt.title("Boxplots of Price")
        plt.show()

        # review rating kernel density
        sns.kdeplot(df['review_rating'])
        plt.title('Kernel Density Plot of Review Rating')
        plt.show()

        # remove incorrectly recorded and outlying values
        df = df[df.price <= 30000]
        df = df[df.bedrooms <= 40]

        # density of observations
        sns.kdeplot(df['longitude'],
                df['latitude'],
                color='r', shade=True,
                cmap="Reds", shade_lowest=False)
        plt.title("Density of AirBnbs in Cape Town")
        plt.show()

        # price 
        df.plot(x='longitude',
                y='latitude',
                color=df['price'].values,
                cmap="jet",
                title='Prices of Airbnbs in Cape Town',
                kind='scatter',
                s= 12)
        plt.title("AirBnb Prices in Cape Town")
        plt.show()

    # continuous reponse to multiclass
    cut_bins = [0, 750, 1450, 6000, 1000000]
    bin_labels = [1, 2, 3, 4]
    df['price'] = pd.cut(df['price'], bins=cut_bins, labels=bin_labels)

    return df

def prepareData(df):

    X = df.drop(columns=['price'])
    Y = (df['price']).ravel()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X)
    X = scaler.transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    dummy_y_train = np_utils.to_categorical(encoded_Y)

    encoder.fit(y_test)
    encoded_Y = encoder.transform(y_test)
    dummy_y_test = np_utils.to_categorical(encoded_Y)

    x_train = x_train.reshape(len(x_train), 15).astype("float32")
    x_test = x_test.reshape(len(x_test), 15).astype("float32")

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    return x_train, y_train, dummy_y_train, x_test, y_test, dummy_y_test

def metrics(Y, Yhat):
    metrics = {
        'Accuracy': accuracy_score(Y, Yhat),
        'Precision': precision_score(Y, Yhat, average='macro',
                                     zero_division=0),
        'Recall': recall_score(Y, Yhat, average='macro', zero_division=0)
    }

    return metrics

def NN(nodes, layers, dropout):
    
    NN = Sequential()
    NN.add(BatchNormalization())
    NN.add(
        Dense(nodes,
              input_dim=15,
              activation='relu',
              kernel_initializer=tf.keras.initializers.RandomNormal(
                  mean=0., stddev=1)))
                  
    NN.add(Dropout(dropout))
    for i in range(layers):
        NN.add(BatchNormalization())
        NN.add(
            Dense(nodes,
                  activation='relu',
                  kernel_initializer=tf.keras.initializers.RandomNormal(
                      mean=0., stddev=1)))
        NN.add(Dropout(dropout))
    NN.add(BatchNormalization())
    NN.add(
        Dense(4,
              activation='softmax',
              kernel_initializer=tf.keras.initializers.RandomNormal(
                  mean=0., stddev=1)))

    return NN

def gridPlots():
    grid_df = pd.read_csv('grid_df_big.csv')
    grid_df.plot(y=['TAccuracy', 'VAccuracy'], color=['blue', 'orange'])
    plt.title('Accuracy during Grid Search')
    plt.xlabel("Grid Search Iteration")
    plt.ylabel("Accuracy")
    plt.axhline(y=0.322672, color='r', linestyle='-')
    plt.axhline(y=0.611)
    plt.show()

    grid_df.plot(y=['TPrecision', 'VPrecision'])
    plt.title('Precision during Grid Search')
    plt.xlabel("Grid Search Iteration")
    plt.ylabel("Precision")
    plt.axhline(y=0.665)
    plt.show()

    grid_df.plot(y=['TRecall', 'VRecall'])
    plt.title('Recall during Grid Search')
    plt.xlabel("Grid Search Iteration")
    plt.ylabel("Recall")
    plt.axhline(y=0.492)
    plt.show()

def optimalModel(X_train, Y_train, X_test, dummy_Y_test, Y_test):
    print("\nLoading model...\n")
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    full_model = NN(20, 2, 0)  

    full_model.compile(loss='categorical_crossentropy',
                                   optimizer=opt,
                                   metrics=[
                                       'categorical_accuracy',
                                       tf.keras.metrics.Precision(),
                                       tf.keras.metrics.Recall()
                                   ])

    history = full_model.fit(X_train,
                                         Y_train,
                                         epochs=60,
                                         batch_size=128,
                                         verbose=False)
    

    # plt.plot(history.history['loss'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')

    # plt.title('Training Loss Per Epoch')
    # plt.show()

    training_score = full_model.evaluate(X_train,
                                                 Y_train,
                                                 verbose=0)
    test_score = full_model.evaluate(X_test,
                                                 dummy_Y_test,
                                                    verbose=0)
    print("Optimal model training metrics:"), print("Loss ; Accuracy ; Precision ; Recall"), print(training_score)
    print("Optimal model test metrics:"), print("Loss ; Accuracy ; Precision ; Recall"), print(test_score)
    print()

    price_predictions = (argmax(full_model.predict(X_test))).numpy()+1

    pred = pd.DataFrame({'Predicted': price_predictions})
    actual = pd.DataFrame({'Acutal': Y_test})
    results = pred.join(actual)
   
    results.to_csv('results.csv')


    return full_model

def evaluate(model):
    
        print(
            "Predict your AirBnb's listing price by filling in the following prompts:"
        )
        pax = input("Accomdates: ")
        bedrooms = input("Bedrooms: ")
        latitude = input("Latitude (Around -33): ")
        longitude = input("Longitude (Around 18): ")
        min_nights = input("Min nights: ")
        max_nights = input("Max nights: ")
        type = input("Private home [P], shared room [S], or hotel room [H]: ")
        if type == 'P':
            entire_home = 1
            hotel_room = 0
            shared_room = 0
        elif type == 'S':
            entire_home = 0
            hotel_room = 0
            shared_room = 1
        elif type == 'H':
            entire_home = 0
            hotel_room = 1
            shared_room = 0
        instant = input("Instantly bookable [Y/N]: ")
        instant = 0 if instant == 'Y' else 1

        X_user = pd.DataFrame([[
            0, pax, bedrooms,
            4.63, latitude, longitude, entire_home,
            hotel_room, shared_room, min_nights, max_nights, 188,
            0, 0, instant
        ]],
                              columns=[
                                  'superhost', 'pax', 'bedrooms',
                                  'review_rating', 'latitude', 'longitude',
                                  'entire_home', 'hotel_room', 'shared_room',
                                  'mininmum_nights', 'maximum_nights',
                                  'availability_365', 'number_of_reviews',
                                  'fullyBooked', 'instant_bookable'
                              ])

        X_user = X_user.astype("float64")

        scaler = MinMaxScaler(feature_range=(-1,1))
        X_user = scaler.fit_transform(X_user)
        price_prediction = argmax(model.predict(X_user)).numpy()[0]
        if (price_prediction == 0):
            print("You should price your property from R0 - R750.")
        elif (price_prediction == 1):
            print("You should price your property from R750 - R1450.")
        elif (price_prediction == 2):
            print("You should price your property from R1450 - R6000.")
        elif (price_prediction == 3):
            print("You should price your property from R6000 - R15000.")
            print("Consult the AirBnb website if you believe your listing is worth more than R15000.")

def main():
  
    # A. INTRODUCTION

    print("Welcome to the")
    print("AirBnb Pricing and Valuation Tool using Neural Networks")
    print("© Fleischer-Gemini 2021")
    print("-" * 40, end="\n"), print()

    # B. DATA PROCESSING AND EXPLORATION

    df = loadData('/Users/jessiebosman/Desktop/AI/A2/airbnbCapeTown.csv')
    df = selectData(df)
    df = cleanData(df)
    df = EDA(df)

    # C. MODEL DESIGN AND TRAINING

    print()
    X_train, Y_train, dummy_Y_train, X_test, Y_test, dummy_Y_test = prepareData(
        df)

    # 1. NAIVE MODEL

    if args.models:
        print("1. NAIVE MODEL")
        print()

        naive_model = naiveModel(df, X_train, dummy_Y_train)
        naive_predictions = pd.Series(
            naive_model.predict(verbose=True, X=X_test))

    # 2. SHALLOW, BASELINE NEURAL NETWORK

    if args.models:
        print("2. BASELINE MODEL")
        print()
        baseline_loss_train = []
        baseline_acc_train = []
        baseline_prec_train = []
        baseline_recall_train = []
        baseline_loss_valid = []
        baseline_acc_valid = []
        baseline_prec_valid = []
        baseline_recall_valid = []

        baseline = NN(5, 2, 0)

        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        i = 1

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
        for train_ix, test_ix in kfold.split(X_train, Y_train):

            baseline.compile(loss='categorical_crossentropy',
                             optimizer=opt,
                             metrics=[
                                 'categorical_accuracy',
                                 tf.keras.metrics.Precision(),
                                 tf.keras.metrics.Recall()
                             ])
            baseline.fit(X_train[train_ix],
                         dummy_Y_train[train_ix],
                         epochs=10,
                         batch_size=256,
                         verbose=False)

            # Generate generalization metrics
            training_score = baseline.evaluate(X_train[train_ix],
                                               dummy_Y_train[train_ix],
                                               verbose=0)
            baseline_loss_train.append(training_score[0])
            baseline_acc_train.append(training_score[1])
            baseline_prec_train.append(training_score[2])
            baseline_recall_train.append(training_score[3])

            validation_score = baseline.evaluate(X_train[test_ix],
                                                 dummy_Y_train[test_ix],
                                                 verbose=0)
            baseline_loss_valid.append(validation_score[0])
            baseline_acc_valid.append(validation_score[1])
            baseline_prec_valid.append(training_score[2])
            baseline_recall_valid.append(training_score[3])

            print(f'{i}...')
            i += 1

        print(), print(
            f"Baseline model training accuracy: {mean(baseline_acc_train)}")
            
        print(f"Basline model validation accuracy: {mean(baseline_acc_valid)}"
              ), print(mean(baseline_prec_valid)), print(mean(baseline_recall_valid)), print()


    
    # 3. DEEPER, FULLY-CONNECTED NEURAL NETWORK
    
    if args.models:
        print("3. FULL MODEL"), print()

        grid_df = pd.DataFrame(columns=[
            'Epochs', 'Layers', 'Nodes', 'Learning Rate', 'Batch Size',
            'Dropout', 'TLoss', "TAccuracy", "TPrecision", "TRecall", 'VLoss',
            "VAccuracy", "VPrecision", "VRecall"
        ])

        if args.basic:
            param_grid = {
                'epochs': [20],
                'layers': [2],
                'nodes': [20],
                'lr': [0.01],
                'batch_size': [64],
                'dropout': [0.0]
            }
        else:
           
            param_grid = {
                'epochs': [20, 40, 60],
                'layers': [2, 3],
                'nodes': [5, 10, 20],
                'lr': [0.1, 0.01, 0.001],
                'batch_size': [32, 128, 256],
               'dropout': [0, 0.1, 0.2]
            }

        grid = ParameterGrid(param_grid)
        f = 1
        hp_train_results = []
        hp_valid_results = []
        for params in grid:
            
            print("Hyperparameters:")
            print(params)
            print(f"{f} of 324")
            f += 1
            print()

            full_model_loss_train = []
            full_model_acc_train = []
            full_model_prec_train = []
            full_model_recall_train = []
            full_model_loss_valid = []
            full_model_acc_valid = []
            full_model_prec_valid = []
            full_model_recall_valid = []

            full_model = NN(params['nodes'], params['layers'],
                            params['dropout'])

            opt = tf.keras.optimizers.Adam(learning_rate=params['lr'])
            i = 1

            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

            for train_ix, test_ix in kfold.split(X_train, Y_train):
                print(f'{i}...')

                full_model.compile(loss='categorical_crossentropy',
                                   optimizer=opt,
                                   metrics=[
                                       'categorical_accuracy',
                                       tf.keras.metrics.Precision(),
                                       tf.keras.metrics.Recall()
                                   ])

                history = full_model.fit(X_train[train_ix],
                                         dummy_Y_train[train_ix],
                                         epochs=params['epochs'],
                                         batch_size=params['batch_size'],
                                         verbose=False)

                

                # Generate generalization metrics
                training_score = full_model.evaluate(X_train[train_ix],
                                                     dummy_Y_train[train_ix],
                                                     verbose=0)
                full_model_loss_train.append(training_score[0])
                full_model_acc_train.append(training_score[1])
                full_model_prec_train.append(training_score[2])
                full_model_recall_train.append(training_score[3])

                validation_score = full_model.evaluate(X_train[test_ix],
                                                       dummy_Y_train[test_ix],
                                                       verbose=0)
                full_model_loss_valid.append(validation_score[0])
                full_model_acc_valid.append(validation_score[1])
                full_model_prec_valid.append(validation_score[2])
                full_model_recall_valid.append(validation_score[3])

                i += 1


            grid_df = grid_df.append(
                {
                    'Epochs': params['epochs'],
                    'Layers': params['layers'],
                    'Nodes': params['nodes'],
                    'Learning Rate': params['lr'],
                    'Batch Size': params['batch_size'],
                    'Dropout': params['dropout'],
                    'TLoss': mean(full_model_loss_train),
                    'TAccuracy': mean(full_model_acc_train),
                    'TPrecision': mean(full_model_prec_train),
                    'TRecall': mean(full_model_recall_train),
                    'VLoss': mean(full_model_loss_valid),
                    'VAccuracy': mean(full_model_acc_valid),
                    'VPrecision': mean(full_model_prec_valid),
                    'VRecall': mean(full_model_recall_valid)
                },
                ignore_index=True)

            hp_train_results.append({
                'Loss': mean(full_model_loss_train),
                'Acc': mean(full_model_acc_train),
                'Prec': mean(full_model_prec_train),
                'Rec': mean(full_model_recall_train)
            })
            hp_valid_results.append({
                'Loss': mean(full_model_loss_valid),
                'Acc': mean(full_model_acc_valid),
                'Prec': mean(full_model_prec_valid),
                'Rec': mean(full_model_recall_valid)
            })

            print()

        r = 0

        grid_df.to_csv("grid_df.csv")
        print()
        for r in range(len(hp_train_results)):
            print(f"{r}.")
            print("Training")
            print(hp_train_results[r])
            print('Validation')
            print(hp_valid_results[r])

            print()
            r += 1


    if args.grids:
        gridPlots()
    
   
    optimal_model = optimalModel(X_train, dummy_Y_train, X_test, dummy_Y_test, Y_test)
    
    # --------------------------------------------------
    # D. TESTING
    # --------------------------------------------------
    if args.predict and args.models:
        print()

        
        naive_metrics = {
                'Accuracy':
                accuracy_score(Y_test, naive_predictions),
                'Precision':
                precision_score(Y_test,
                                naive_predictions,
                                average='macro',
                                zero_division=0),
                'Recall':
                recall_score(Y_test,
                             naive_predictions,
                             average='macro',
                             zero_division=0)
            }

        print("a. Naive Model Test Classification Metrics")
        print("-" * 40, end="\n")
        print(naive_metrics)
        print()

       
        baseline_predictions = [
                (argmax(probabilities).numpy() + 1)
                for probabilities in baseline.predict(X_test)
            ]

        baseline_metrics = metrics(Y_test, baseline_predictions)

        print("b. Baseline Model Test Classification Metrics")
        print("-" * 40, end="\n")
        print(baseline_metrics)
        print()
        
        
        full_predictions = [(argmax(probabilities).numpy() + 1)
                                for probabilities in optimal_model.predict(X_test)
                                ]

        fullmodel_metrics = metrics(Y_test, full_predictions)
        print("c. Full Model Test Classification Metrics")
        print("-" * 40, end="\n")
        print(fullmodel_metrics)
        print()

        # plt.show()
    elif (args.predict):
        print("You must select a model to predict with. See --help")

    # --------------------------------------------------
    # E. TESTING
    # --------------------------------------------------
    if args.predict or args.evaluate:
        evaluate(optimal_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neural Networks for AirBnb Pricing")
    parser.add_argument(
        '--models',
        action='store_true',
        help="Run the entire program, including EDA and training. This populates the hyperparameter grid search file."
    )
    parser.add_argument(
        '--basic',
        action='store_true',
        help="Run the entire program, including EDA, training and predictions - with basic hyperparameter search.")
    parser.add_argument(
        '--predict',
        action='store_true',
        help=
        "Include the test metrics of the models and evaluate an individual entry."
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help=
        "Run the optimal model and evaluate an individual entry"
    )
    
    parser.add_argument(
        '--grids',
        action='store_true',
        help = "View the hyperparameter grid search plots."
    )
    args = parser.parse_args()
    main()