# Feed-Forward Neural Networks as Price Predictors on Airbnb
## Antony Fleischer 

### Install Requirements
To install the packages for this project, run:

    pip3 install -r requirements.txt

### Running the Program
To view all of the options for running this program, run:

    python3 airbnbNet.py --help

However, some good use cases are:
1. Run the full program as above, but with a basic hyperparameter grid search. This just demonstrates functionality, but I would use it to test my work.

        python3 airbnbNet.py --models --basic --predict --evaluate
        
2. Run the full program: exploratory data analysis, training, hyperparameter grid search, predictions and evalauation of a custom listing. [This will take very, very long]

        python3 airbnbNet.py --models --predict --evaluate

2
3. View the plots from the hyperparameter grid search.

        python3 airbnbNet.py --grids

4. Evaluate a custom listing. [The airbnb pricing tool.]

        python3 airbnbNet.py --evaluate
