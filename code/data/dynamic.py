import pandas as pd
import numpy as np
import torch
import os
import random

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from .utils import normalize_score, METRICS_OF_INTEREST, BBLITE_TASKS, BBHARD_SUBTASKS
from .dynamic_utils import prepare_dataset_from_df, prepare_dataset_from_df_for_cf

prepare_funcs = {
    "mlp": prepare_dataset_from_df,
    "mlp_for_search": prepare_dataset_from_df,
    "svd": prepare_dataset_from_df_for_cf,
    "model_model_knn": prepare_dataset_from_df_for_cf,
    "task_task_knn": prepare_dataset_from_df_for_cf,
}

def preprocess(args, logger, full_file, selected_tasks):
    preprocess_funcs = {
        "l0": preprocess_l0, # a pre-defined fixed train/dev/test split, for quick prototyping and maybe for search
        "l1": preprocess_l1, # if there are k model families, do k-fold, by rotating who is dev/test, for search
        "l2": preprocess_l2, # if there are k model families, do a k(k-1)-fold nested cv, for final eval
    }

    assert args.search_cv_level in preprocess_funcs.keys()

    ret = preprocess_funcs[args.search_cv_level](args, logger, full_file, selected_tasks)

    return ret

def preprocess_l0(args, logger, full_file, selected_tasks):
    df = pd.read_csv(full_file, index_col=False)

    if args.preferred:
        df = df[df["is_preferred_metric"] == 1].reset_index()

    df = df[df["metric_name"].isin(METRICS_OF_INTEREST)]
    df = df.reset_index(drop=True)

    dev_model = "GPT"
    test_model = "Gopher" # actually it will not be used during search

    logger.info("Dev Model: {}, Test Model: {}".format(dev_model, test_model))

    df_temp = df.copy()
    test_mask = df_temp.apply(lambda row: (row['subtask']) not in selected_tasks and (row["model_family"]==test_model), axis=1)
    dev_mask = df_temp.apply(lambda row: (row['subtask']) not in selected_tasks and (row["model_family"]==dev_model), axis=1)

    test_df = df_temp[test_mask].reset_index()
    dev_df = df_temp[dev_mask].reset_index()
    train_mask = ~(dev_mask | test_mask)
    train_df = df_temp[train_mask].reset_index()

    all_sets = []
    all_sets.append(
        prepare_funcs[args.search_subroutine](logger, train_df, dev_df, test_df)
    )    

    return all_sets # actually just one item, but just to be consitent across l0/l1/l2

def preprocess_l1(args, logger, full_file, selected_tasks):
    df = pd.read_csv(full_file, index_col=False)

    if args.preferred:
        df = df[df["is_preferred_metric"] == 1].reset_index()

    df = df[df["metric_name"].isin(METRICS_OF_INTEREST)]
    df = df.reset_index(drop=True)

    model_keys = list(df.groupby(["model_family"]).groups.keys())
    random.Random(42).shuffle(model_keys) # This should fix things...?

    all_sets = []

    for model_fold_idx in range(len(model_keys)):
        test_model = model_keys[model_fold_idx]
        dev_model = model_keys[model_fold_idx-1]
        logger.info("Dev Model: {}, Test Model: {}".format(dev_model, test_model))

        df_temp = df.copy()
        test_mask = df_temp.apply(lambda row: (row['subtask']) not in selected_tasks and (row["model_family"]==test_model), axis=1)
        dev_mask = df_temp.apply(lambda row: (row['subtask']) not in selected_tasks and (row["model_family"]==dev_model), axis=1)

        test_df = df_temp[test_mask].reset_index()
        dev_df = df_temp[dev_mask].reset_index()
        train_mask = ~(dev_mask | test_mask)
        train_df = df_temp[train_mask].reset_index()

        all_sets.append(
            prepare_funcs[args.search_subroutine](logger, train_df, dev_df, test_df)
        )

    # should be a list of 6 datasets (#model family=6), 
    # each one is (X_train, y_train, X_dev, y_dev, X_test, y_test, (encoder, scaler))
    return all_sets
    
def preprocess_l2(args, logger, full_file, selected_tasks):
    df = pd.read_csv(full_file, index_col=False)

    if args.preferred:
        df = df[df["is_preferred_metric"] == 1].reset_index()

    df = df[df["metric_name"].isin(METRICS_OF_INTEREST)]
    df = df.reset_index(drop=True)

    model_keys = list(df.groupby(["model_family"]).groups.keys())
    random.Random(42).shuffle(model_keys) # This should fix things...?

    all_sets = []

    for dev_model_idx in range(len(model_keys)):
        for test_model_idx in range(len(model_keys)):

            if dev_model_idx == test_model_idx:
                continue

            test_model = model_keys[test_model_idx]
            dev_model = model_keys[dev_model_idx]
            logger.info("Dev Model: {}, Test Model: {}".format(dev_model, test_model))

            df_temp = df.copy()
            test_mask = df_temp.apply(lambda row: (row['subtask']) not in selected_tasks and (row["model_family"]==test_model), axis=1)
            dev_mask = df_temp.apply(lambda row: (row['subtask']) not in selected_tasks and (row["model_family"]==dev_model), axis=1)

            test_df = df_temp[test_mask].reset_index()
            dev_df = df_temp[dev_mask].reset_index()
            train_mask = ~(dev_mask | test_mask)
            train_df = df_temp[train_mask].reset_index()

            all_sets.append(
                prepare_funcs[args.search_subroutine](logger, train_df, dev_df, test_df)
            )

    # should be a list of 30 datasets (#model family=6, 30 different permutations), 
    # each one is (X_train, y_train, X_dev, y_dev, X_test, y_test, (encoder, scaler))
    return all_sets
