import pandas as pd

from data.dynamic import preprocess
from models.mlp import run_mlp_regression
from .fixed_lists import FIXED_LISTS

def search_run_fixed(args, logger, run_func, selected_tasks=None):
    # read the list of selected tasks from args
    # evaluate this selection and return the scores
    
    if selected_tasks is None:
        # if it is not passed in in the arg, read from the pre-registered name lists
        selected_tasks = FIXED_LISTS[args.selected_tasks]
        logger.info("Selected tasks: {}".format(selected_tasks))
    
    # get data
    datasets = preprocess(args, logger, args.full_file, selected_tasks)
    
    df = pd.DataFrame(columns=["id", "dev_rmse", "dev_r2", "test_rmse", "test_r2", "dev_model_family", "test_model_family"])

    # for each split run it
    # may be possible to parallize this... ? or parallelize search part?
    for i, dataset in enumerate(datasets):
        X_train, y_train, X_dev, y_dev, X_test, y_test, _ = dataset
        (dev_rmse, dev_r2, dev_pred), (test_rmse, test_r2, test_pred) = run_func(args, logger, X_train, y_train, X_dev, y_dev, X_test, y_test)
        
        # TODO(yeqy): need so save prediction in some way ...
        df.loc[len(df.index)] = [i, dev_rmse, dev_r2, test_rmse, test_r2, "", ""]
        
    # get avg performance

    mean_values = df.mean(numeric_only=True)
    std_values = df.std(numeric_only=True)

    df.loc[len(df)] = ['mean'] + list(mean_values)[1:] + ["", ""]
    df.loc[len(df)] = ['std'] + list(std_values)[1:] + ["", ""]

    logger.info(df.drop(columns=["dev_model_family", "test_model_family"]).tail(2).to_string(index=False))

    return mean_values["dev_r2"] # only need this as selection criteria