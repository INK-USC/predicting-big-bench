import torch
import numpy as np

from .mlp_models import MLPRegression, MultiTargetMLPRegression, train, eval_, train_for_search, eval_for_search
from .utils import generate_random_combinations, tensorize

def run_mlp_regression(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test):
    # input_dim, hidden_dims, dropout
    X_train, y_train, X_dev, y_dev, X_test, y_test = tensorize(X_train, y_train, X_dev, y_dev, X_test, y_test)
    args.input_dim = X_train.shape[1]

    tuned_hps = eval(args.hp) if args.hp is not None else {}
    for key in tuned_hps.keys():
        setattr(args, key, tuned_hps[key])

    if args.model_arch == "mlp":
        model = MLPRegression(args.input_dim, args.hidden_dims, args.dropout)
    elif args.model_arch == "multitarget_mlp":
        # a hack to get the number of metrics...
        n_target = int(torch.max(X_train[:, -1]).item()) + 1
        model = MultiTargetMLPRegression(args.input_dim - 1, args.hidden_dims, args.dropout, n_target)

    train(args, logger, model, X_train, y_train, X_dev, y_dev)

    dev_rmse, dev_r2, dev_pred, test_rmse, test_r2, test_pred = 1e10, -1, None, 1e10, -1, None

    if X_dev is not None:
        dev_rmse, dev_r2, dev_pred = eval_(model, X_dev, y_dev)
        logger.info("[Dev] dev_rmse: {:.4f}, r2_score: {:.4f}".format(dev_rmse, dev_r2))

    if X_test is not None:
        test_rmse, test_r2, test_pred = eval_(model, X_test, y_test)
        logger.info("[Test] test_rmse: {:.4f}, r2_score: {:.4f}".format(test_rmse, test_r2))

    return (dev_rmse, dev_r2, dev_pred), (test_rmse, test_r2, test_pred)

def tune_mlp_regression(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test):
    X_train, y_train, X_dev, y_dev, X_test, y_test = tensorize(X_train, y_train, X_dev, y_dev, X_test, y_test)

    param_distributions = {
        "lr": [1e-3, 3e-4, 1e-4],
        "batch_size": [32, 64, 128],
        "dropout": [0.0, 0.05, 0.1, 0.15, 0.2],
        "hidden_dims": [
            (128,64,32,16),(256,128,64,32),
            (128,64,32),(256,128,64),
            (64,32),(128,64),(256,128),
            (128,),(64,)
        ],
        "weight_decay": [0.0, 0.00001, 0.0001, 0.001, 0.01],
    }

    best_dev_rmse, best_r2_score, best_hp_setting = 1e10, -1, None
    for hp_setting in generate_random_combinations(param_distributions, n_trials=args.n_trials):
        for key in hp_setting.keys():
            setattr(args, key, hp_setting[key])

        args.input_dim = X_train.shape[1]

        if args.model_arch == "mlp":
            model = MLPRegression(args.input_dim, args.hidden_dims, args.dropout)
        elif args.model_arch == "multitarget_mlp":
            # a hack to get the number of metrics...
            n_target = int(torch.max(X_train[:, -1]).item()) + 1
            model = MultiTargetMLPRegression(args.input_dim - 1, args.hidden_dims, args.dropout, n_target)

        dev_rmse, dev_r2_score = train(args, logger, model, X_train, y_train, X_dev, y_dev)

        if dev_r2_score > best_r2_score:
            best_r2_score = dev_r2_score
            best_hp_setting = hp_setting

        logger.info("HPs: {}, dev_R2: {:4f}".format(hp_setting, dev_r2_score))
    
    return best_r2_score, best_hp_setting, None


def run_mlp_regression_for_search(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test):
    # input_dim, hidden_dims, dropout
    X_train, y_train, X_dev, y_dev, X_test, y_test = tensorize(X_train, y_train, X_dev, y_dev, X_test, y_test)
    args.input_dim = X_train.shape[1]

    tuned_hps = eval(args.hp) if args.hp is not None else {}
    for key in tuned_hps.keys():
        setattr(args, key, tuned_hps[key])

    if args.model_arch == "mlp":
        model = MLPRegression(args.input_dim, args.hidden_dims, args.dropout)
    elif args.model_arch == "multitarget_mlp":
        # a hack to get the number of metrics...
        n_target = int(torch.max(X_train[:, -1]).item()) + 1
        model = MultiTargetMLPRegression(args.input_dim - 1, args.hidden_dims, args.dropout, n_target)

    best_dev_rmse, best_dev_r2 = train_for_search(args, logger, model, X_train, y_train, X_dev, y_dev)
    logger.info("[Dev] dev_rmse: {:.4f}, r2_score: {:.4f}".format(best_dev_rmse, best_dev_r2))

    return (best_dev_rmse, best_dev_r2, None), (0.0, 0.0, None)