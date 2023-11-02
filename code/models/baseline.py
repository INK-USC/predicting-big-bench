from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import Dataset, accuracy, Reader

from .surprise_utils import surprise_eval, CustomFold, reorder_surprise_preds

def run_baseline(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test):
    assert args.model_arch in ["random", "bsl_model", "bsl_task", "bsl_model_task"]

    algo = NormalPredictor() if args.model_arch == "random" else BaselineOnly()

    reader = Reader(rating_scale=(0, 1))
    train_data = Dataset.load_from_df(X_train[["model_unique", "task_unique", "score"]], reader)


    trainset = train_data.build_full_trainset()
    algo.fit(trainset)

    dev_rmse, dev_r2, dev_pred, test_rmse, test_r2, test_pred = 1e10, -1, None, 1e10, -1, None

    # in doc (https://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly)
    # If user is unknown, then the bias is assumed to be zero. The same applies for item 
    # So i just set them to be something unseen
    if X_dev is not None:
        if args.model_arch == "bsl_model":
            X_dev["task_unique"] = "unseen_task"
        elif args.model_arch == "bsl_task":
            X_dev["model_unique"] = "unseen_model"
        dev_data = Dataset.load_from_df(X_dev[["model_unique", "task_unique", "score"]], reader)
        devset = dev_data.build_full_trainset().build_testset()
        predictions = algo.test(devset)
        dev_rmse, dev_r2, dev_pred = surprise_eval(predictions)
        dev_pred = reorder_surprise_preds(X_dev, predictions)

    if X_test is not None:
        if args.model_arch == "bsl_model":
            X_test["task_unique"] = "unseen_task"
        elif args.model_arch == "bsl_task":
            X_test["model_unique"] = "unseen_model"
        test_data = Dataset.load_from_df(X_test[["model_unique", "task_unique", "score"]], reader)
        testset = test_data.build_full_trainset().build_testset()
        predictions = algo.test(testset)
        test_rmse, test_r2, test_pred = surprise_eval(predictions)
        test_pred = reorder_surprise_preds(X_test, predictions)

    return (dev_rmse, dev_r2, dev_pred), (test_rmse, test_r2, test_pred)
