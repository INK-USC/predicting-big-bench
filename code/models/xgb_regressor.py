import numpy as np
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import r2_score, mean_squared_error

def run_xgb_regressor(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test):

    tuned_parameters = eval(args.hp) if args.hp is not None else {}
  
    regr = xgb.XGBRegressor(
        objective="reg:squarederror", 
        random_state=42, 
        nthread=5,
        early_stopping_rounds=5,
        **tuned_parameters
    )

    if X_dev is not None:
        regr.fit(X_train, y_train, eval_set=[(X_dev, y_dev)])
        print(regr.best_score)
    else:
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

def tune_xgb_regressor(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test):
    assert (X_dev is not None) and (y_dev is not None)

    # To use scikit-learn's HP search, we need to create a manual data split as follows
    X_combined = np.concatenate((X_train, X_dev), axis=0)
    y_combined = np.concatenate((y_train, y_dev))

    # -1 means will always be in train set; 0 means it's in the 0-th dev set
    split_index = [-1] * X_train.shape[0] + [0] * X_dev.shape[0] 
    ps = PredefinedSplit(test_fold=split_index)

    # define model
    regr = xgb.XGBRegressor(
        objective="reg:squarederror", 
        random_state=42,
        verbosity=0,
        early_stopping_rounds=5,
        nthread=10)

    # define search space
    param_distributions = {
        "n_estimators": [30, 100, 300, 1000],
        "learning_rate": [0.1, 0.3, 0.5, 0.8, 1.0],
        "max_depth": [None, 16, 32, 64, 128], # None indicates no limit
        "gamma": [0, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    clf = RandomizedSearchCV(regr, param_distributions, random_state=0, n_iter=args.n_trials, cv=ps, verbose=5, n_jobs=1)
    search = clf.fit(X_combined, y_combined, eval_set=[(X_dev, y_dev)], verbose=False)

    return search.best_score_, search.best_params_, search.cv_results_