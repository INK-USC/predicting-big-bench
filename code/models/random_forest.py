import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

def run_random_forest(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test):

    tuned_hps = eval(args.hp) if args.hp is not None else {}

    regr = RandomForestRegressor(
        random_state=42, 
        criterion="squared_error",
        **tuned_hps,
    )
    regr.fit(X_train, y_train)

    # set some default value
    dev_rmse, dev_r2, dev_pred, test_rmse, test_r2, test_pred = 1e10, -1, None, 1e10, -1, None

    # dev
    if X_dev is not None:
        dev_pred = regr.predict(X_dev)
        dev_mse = mean_squared_error(y_true=y_dev, y_pred=dev_pred) # np.mean((y_pred - y_dev)**2)
        dev_rmse = np.sqrt(dev_mse)
        dev_r2 = r2_score(y_true=y_dev, y_pred=dev_pred)

    # test
    if X_test is not None:
        test_pred = regr.predict(X_test)
        test_mse = mean_squared_error(y_true=y_test, y_pred=test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_true=y_test, y_pred=test_pred)

    return (dev_rmse, dev_r2, dev_pred), (test_rmse, test_r2, test_pred)

def tune_random_forest(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test):
    assert (X_dev is not None) and (y_dev is not None)

    # To use scikit-learn's HP search, we need to create a manual data split as follows
    X_combined = np.concatenate((X_train, X_dev), axis=0)
    y_combined = np.concatenate((y_train, y_dev))

    # -1 means will always be in train set; 0 means it's in the 0-th dev set
    split_index = [-1] * X_train.shape[0] + [0] * X_dev.shape[0] 
    ps = PredefinedSplit(test_fold=split_index)

    # define model
    regr = RandomForestRegressor(criterion="squared_error")
    # define search space
    param_distributions = {
        "n_estimators": [30, 100, 300],
        "max_depth": [None, 16, 32, 64, 128],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, 0.9, 0.8, 0.7, 0.6, "sqrt"],
        "max_samples": [1.0, 0.9, 0.8, 0.7, 0.6]
    }

    clf = RandomizedSearchCV(regr, param_distributions, cv=ps, 
        random_state=42, n_iter=args.n_trials, verbose=5, n_jobs=4
    )
    search = clf.fit(X_combined, y_combined)

    return search.best_score_, search.best_params_, search.cv_results_