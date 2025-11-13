import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
import joblib

def data_cleaning():
    # import as csv
    data = pd.read_csv('data-climbing.csv')

    # remove unnecessary fields
    data = data.drop(['Timestamp'], axis=1)
    data = data.drop(['Name'], axis=1)
    data = data.drop(['Email Address'], axis=1)

    # compute weighted pullup as percentage: derived attributed
    data['percentage_pullup'] = -1
    data['percentage_pullup'] = data['wp'] / data['body_weight']

    data = data.drop(['body_weight'], axis = 1)
    data = data.drop(['wp'], axis = 1)

    ## turn motivation into number
    # data['motivator_num'] = -1
    # for idx, row in data.iterrows():
    #     if row['motivation'] == 'for the joy of it':
    #         data.loc[idx, 'motivator_num'] = 0
    #     elif row['motivation'] == 'to be strong / as a form of exercise':
    #         data.loc[idx, 'motivator_num'] = 1
    #     elif row['motivation'] == 'socialize / community':
    #         data.loc[idx, 'motivator_num'] = 2
    #     elif row['motivation'] == 'emotional support':
    #         data.loc[idx, 'motivator_num'] = 3
    ## one-hot encode motivation
    data = pd.get_dummies(data, columns=['motivation'], prefix='motivation')

    #data = data.drop(['motivation'], axis=1)
    data['experience'] = data['experience'].str.split(" ").str[0].astype(float)

    # pull out labels
    data['grade'] = data['grade'].astype(str).str[1:] # remove "V"
    y = data[['grade']].astype(float)
    X = data.drop(['grade'], axis = 1)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")
    print(X.columns.tolist())
    return X_scaled, y

def elasticNet_train_test():   
    X, y = data_cleaning()
    
    # create & train model
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)

    param_grid = {
        'alpha': np.logspace(-3, 1, 10),  # Regularization strength
        'l1_ratio': np.linspace(0.1, 0.9, 9)  # Mix between Lasso (L1) and Ridge (L2)
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)

    # Best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, "elasticNet_train_test.pkl")
    return best_model

def elasticNet_kfcv():
    X, y = data_cleaning()

    # Define Elastic Net and hyperparameter grid
    elastic_net = ElasticNet()

    param_grid = {
        'alpha': np.logspace(-3, 1, 10),  # Regularization strength
        'l1_ratio': np.linspace(0.1, 0.9, 9)  # Mix of L1 (Lasso) and L2 (Ridge)
    }

    # Use K-Fold Cross-Validation since we have a small dataset
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(elastic_net, param_grid, cv=kf, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)

    # Best model
    best_model = grid_search.best_estimator_
    
    joblib.dump(best_model, "elasticNet_kfcv.pkl")

    return best_model

def xgboost_regression():
    from xgboost import XGBRegressor

    X, y = data_cleaning()
    
    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [1, 2, 3]
    }

    # Initialize XGBoost model
    model = XGBRegressor()

    # Define cross-validation strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)  # Ensure `X_train_scaled` is preprocessed

    # Get the best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, "xgboost_regression.pkl")

    return best_model

data_cleaning()
elasticNet_train_test()
elasticNet_kfcv()
xgboost_regression()


