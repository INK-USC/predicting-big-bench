import os
import pandas as pd

from data import load_data

from models.random_forest import run_random_forest, tune_random_forest
from models.xgb_regressor import run_xgb_regressor, tune_xgb_regressor
from models.mlp import run_mlp_regression, tune_mlp_regression, run_mlp_regression_for_search
from models.svd import run_svd, tune_svd
from models.baseline import run_baseline
from models.knn import run_knn, tune_knn

from search.fixed import search_run_fixed
from search.random import search_random
from search.beam_search import search_beam_search
from search.genetic_algorithm import search_genetic_algorithm
from search.simulated_annealing import search_simulated_annealing
from search.batch import search_run_batch

from utils import save_predictions, seed_everything

run_funcs = {
    "random_forest": run_random_forest,
    "xgb": run_xgb_regressor,
    "mlp": run_mlp_regression,
    "multitarget_mlp": run_mlp_regression,
    "svd": run_svd,
    "svdpp": run_svd,
    "random": run_baseline,
    "bsl_model": run_baseline,
    "bsl_task": run_baseline,
    "bsl_model_task": run_baseline,
    "task_task_knn": run_knn,
    "model_model_knn": run_knn,
    "mlp_for_search": run_mlp_regression_for_search, # a faster version for mlp training designed for search
}

tune_funcs = {
    "random_forest": tune_random_forest,
    "xgb": tune_xgb_regressor,
    "mlp": tune_mlp_regression,
    "multitarget_mlp": tune_mlp_regression,
    "svd": tune_svd,
    "svdpp": tune_svd,
    "task_task_knn": tune_knn,
    "model_model_knn": tune_knn
}

search_funcs = {
    "fixed": search_run_fixed,
    "random": search_random,
    "beam": search_beam_search,
    "batch": search_run_batch,
    "ga": search_genetic_algorithm,
    "sa": search_simulated_annealing
}

def run(args, logger):

    if args.mode == "tunehp" or args.mode == "tunehp_then_multi_run":
        seed_everything(args.seed)
        # load data
        X_train, y_train, X_dev, y_dev, X_test, y_test, others = load_data(args, logger, args.data_mode)

        func = tune_funcs[args.model_arch]
        best_r2, best_hp_setting, cv_results = func(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test)

        logger.info("Best R2: {:.4f}, Best HPs: {}".format(best_r2, best_hp_setting))
        if args.save_tuning_logs and cv_results is not None:
            df = pd.DataFrame(cv_results)
            df.to_csv(os.path.join(args.output_dir, "tuning.csv"), index=False)

        # mainly used when args.mode == "tunehp_then_multi_run"; Using str() for being lazy...
        args.hp = str(best_hp_setting) 

    if args.mode == "run":
        seed_everything(args.seed)
        # load data
        X_train, y_train, X_dev, y_dev, X_test, y_test, others = load_data(args, logger, args.data_mode)

        func = run_funcs[args.model_arch]
        (dev_rmse, dev_r2, dev_pred), (test_rmse, test_r2, test_pred) = func(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test)
        
        logger.info("Dev RMSE: {:.4f}; Dev R2: {:.4f}; Test RMSE: {:.4f}; Test R2: {:.4f}".format(
            dev_rmse, dev_r2, test_rmse, test_r2))

        if args.save_predictions:
            if dev_pred is not None:
                save_predictions(args.dev_file, dev_pred, identifier=args.model_arch)
            if test_pred is not None:
                save_predictions(args.test_file, test_pred, identifier=args.model_arch)

    if args.mode == "multi_run" or args.mode == "tunehp_then_multi_run":
        assert args.data_dir is not None
        # data looks like data_dir/0/{train|dev|test}.csv
        seed_everything(args.seed)

        df = pd.DataFrame(columns=["id", "dev_rmse", "dev_r2", "test_rmse", "test_r2", "data_directory"])

        # folds = os.listdir(args.data_dir)
        subdirectories = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]
        folds = [subdir for subdir in subdirectories if subdir.split("/")[-1].isdigit()]
        # folds = [os.path.join(args.data_dir, "0")]

        for i, fold in enumerate(folds):
            # fold_dir = os.path.join(args.data_dir, fold)
            args.train_file = os.path.join(fold, "train.csv")
            args.dev_file = os.path.join(fold, "dev.csv")
            args.test_file = os.path.join(fold, "ctest.csv") if args.controlled_test else os.path.join(fold, "test.csv")

            # load data
            X_train, y_train, X_dev, y_dev, X_test, y_test, others = load_data(args, logger, args.data_mode)
            func = run_funcs[args.model_arch]
            (dev_rmse, dev_r2, dev_pred), (test_rmse, test_r2, test_pred) = func(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test)
            
            logger.info("[Run {}/{}] Dev RMSE: {:.4f}; Dev R2: {:.4f}; Test RMSE: {:.4f}; Test R2: {:.4f}".format(
                i, len(folds),
                dev_rmse, dev_r2, test_rmse, test_r2)
            )

            df.loc[len(df.index)] = [i, dev_rmse, dev_r2, test_rmse, test_r2, fold]

            if args.save_predictions:
                if dev_pred is not None:
                    save_predictions(args.dev_file, dev_pred, identifier=args.model_arch)
                if test_pred is not None:
                    save_predictions(args.test_file, test_pred, identifier=args.model_arch)
        
        mean_values = df.mean(numeric_only=True)
        std_values = df.std(numeric_only=True)

        df.loc[len(df)] = ['mean'] + list(mean_values)[1:] + [""]
        df.loc[len(df)] = ['std'] + list(std_values)[1:] + [""]
        if args.controlled_test:
            df.to_csv(os.path.join(args.output_dir, "multi_run_result_controlled.csv"))
        else:
            df.to_csv(os.path.join(args.output_dir, "multi_run_result.csv"))

        print(df.drop(columns=["data_directory"]).tail(2).to_string(index=False))

    if args.mode == "search":
        seed_everything(args.seed)
        run_func = run_funcs[args.search_subroutine]

        if args.search_mode != "fixed":
            func = search_funcs[args.search_mode]
            best_r2, best_task_list, search_log_df = func(args, logger, run_func)
        else:
            best_r2, best_task_list, search_log_df = -1e10, None, None # will read from args.selected_tasks
        
        if args.save_search_logs and search_log_df is not None:
            search_log_df.to_csv(os.path.join(args.output_dir, "search.csv"), index=False)

        func = search_funcs["batch"]
        result = func(args, logger, run_func, best_task_list)
        