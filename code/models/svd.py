import numpy as np
import pandas as pd

from surprise import Dataset, SVD, SVDpp, accuracy, Reader
from surprise.model_selection.search import RandomizedSearchCV, GridSearchCV

from .surprise_utils import surprise_eval, CustomFold, reorder_surprise_preds

def run_svd(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test):
    # hack: X_train should be the full train df

    reader = Reader(rating_scale=(0, 1))
    train_data = Dataset.load_from_df(X_train[["model_unique", "task_unique", "score"]], reader)

    tuned_hps = eval(args.hp) if args.hp is not None else {}
    algo = SVD(random_state=42, **tuned_hps) if args.model_arch == "svd" else SVDpp(random_state=42, **tuned_hps)

    trainset = train_data.build_full_trainset()
    algo.fit(trainset)

    dev_rmse, dev_r2, dev_pred, test_rmse, test_r2, test_pred = 1e10, -1, None, 1e10, -1, None

    if X_dev is not None:
        dev_data = Dataset.load_from_df(X_dev[["model_unique", "task_unique", "score"]], reader)
        devset = dev_data.build_full_trainset().build_testset()
        predictions = algo.test(devset)
        dev_rmse, dev_r2, dev_pred = surprise_eval(predictions)
        dev_pred = reorder_surprise_preds(X_dev, predictions)

    if X_test is not None:
        test_data = Dataset.load_from_df(X_test[["model_unique", "task_unique", "score"]], reader)
        testset = test_data.build_full_trainset().build_testset()
        predictions = algo.test(testset)
        test_rmse, test_r2, test_pred = surprise_eval(predictions)
        test_pred = reorder_surprise_preds(X_test, predictions)

    return (dev_rmse, dev_r2, dev_pred), (test_rmse, test_r2, test_pred)

def tune_svd(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test):
    assert (X_dev is not None)

    reader = Reader(rating_scale=(0, 1))
    train_data = Dataset.load_from_df(X_train[["model_unique", "task_unique", "score"]], reader)
    dev_data = Dataset.load_from_df(X_dev[["model_unique", "task_unique", "score"]], reader)

    param_grid = {
            'n_factors': [20, 50, 100, 200],
            'n_epochs': [10, 20, 30],
            'lr_all': [0.002, 0.005, 0.007, 0.01],
            'reg_all': [0.00, 0.02, 0.04, 0.06]
    }

    algo = SVD if args.model_arch == "svd" else SVDpp

    # because SVDs are fast so just use grid search directly (SVD++ search takes ~3h)
    clf = GridSearchCV(algo, param_grid, cv=CustomFold(), measures=["rmse"], n_jobs=5, joblib_verbose=100)
    clf.fit((train_data, dev_data))

    print(clf.best_score["rmse"])
    print(clf.best_params["rmse"])

    # convert rmse to r2
    y_mean = np.mean(X_dev["score"])
    base = np.mean((X_dev["score"]- y_mean) ** 2)
    r2_score =  1 - (clf.best_score["rmse"] * clf.best_score["rmse"] / base)

    return r2_score, clf.best_params["rmse"], clf.cv_results