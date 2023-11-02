import os
import pandas as pd
import numpy as np

from data.dynamic import preprocess
from models.mlp import run_mlp_regression
from .fixed_lists import FIXED_LISTS

def search_run_batch(args, logger, run_func, selected_tasks=None):
    # read the list of selected tasks from args
    # evaluate this selection and return the scores
    
    if selected_tasks is None:
        if args.selected_tasks.endswith(".csv"):
            df = pd.read_csv(args.selected_tasks, index_col=None)
            selected_tasks = df["task_list"].values.tolist()
            logger.info("Running {} different task selections ...".format(len(selected_tasks)))
            logger.info("Run 0: {}".format(selected_tasks[0]))
        else:
            # if it is not passed in in the arg, read from the pre-registered name lists
            selected_tasks = [FIXED_LISTS[args.selected_tasks]]
            logger.info("Selected tasks: {}".format(selected_tasks[0]))
    else:
        selected_tasks = [selected_tasks]

    master_df = pd.DataFrame(columns=["run_id", "cv_id", "dev_rmse", "dev_r2", "test_rmse", "test_r2", "task_list"])
    master_summary_df = pd.DataFrame(columns=["run_id", "dev_rmse", "dev_r2", "test_rmse", "test_r2", "n_task", "task_list"])

    for j, selected_tasks_candidate in enumerate(selected_tasks):
        df = pd.DataFrame(columns=["run_id", "cv_id", "dev_rmse", "dev_r2", "test_rmse", "test_r2", "task_list"])
        datasets = preprocess(args, logger, args.full_file, selected_tasks_candidate)
        for i, dataset in enumerate(datasets):
            X_train, y_train, X_dev, y_dev, X_test, y_test, o = dataset
            (dev_rmse, dev_r2, dev_pred), (test_rmse, test_r2, test_pred) = run_func(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test)
            
            # save run results
            df.loc[len(df.index)] = [j, i, dev_rmse, dev_r2, test_rmse, test_r2, selected_tasks_candidate]
            master_df.loc[len(master_df.index)] = [j, i, dev_rmse, dev_r2, test_rmse, test_r2, selected_tasks_candidate]
            
            if args.rebuttal:
                assert len(selected_tasks) == 1 and len(datasets) == 1

                df_train, df_dev, df_test = o[2], o[3], o[4]
                df_dev["predicted_score"] = dev_pred 
                df_dev["diff"] = np.abs(df_dev["normalized_score"] - dev_pred)

                df_test["predicted_score"] = test_pred 
                df_test["diff"] = np.abs(df_test["normalized_score"] - test_pred)

                df_dev.to_csv(os.path.join(args.output_dir, "dev_pred.csv"))
                df_test.to_csv(os.path.join(args.output_dir, "test_pred.csv"))
                df_train.to_csv(os.path.join(args.output_dir, "train.csv"))

        mean_values = df.mean(numeric_only=True)
        # std_values = df.std(numeric_only=True)

        master_summary_df.loc[len(master_summary_df.index)] = [j] + list(mean_values)[2:6] + [len(selected_tasks_candidate), selected_tasks_candidate]

    master_df.to_csv(os.path.join(args.output_dir, "results_full.csv"), index=False)
    master_summary_df.to_csv(os.path.join(args.output_dir, "results_summary.csv"), index=False)
    
