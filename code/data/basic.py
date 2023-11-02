import pandas as pd
import numpy as np
import torch
import os
import pickle

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from .utils import normalize_score, METRICS_OF_INTEREST

def preprocess(args, logger, train_file, dev_file, test_file):
    df = pd.read_csv(train_file, index_col=False)

    if args.preferred:
        df = df[df["is_preferred_metric"] == 1].reset_index()

    df = df[df["metric_name"].isin(METRICS_OF_INTEREST)]
    df = df.reset_index()

    # categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_data = encoder.fit_transform(df[['task', 'subtask', 'model_family', 'model_name', 'metric_name']])

    with open(os.path.join(args.output_dir, "feature_names.pkl"), "wb") as file_obj:
        pickle.dump(encoder.categories_, file_obj)

    # numerical features
    df["log_non_embedding_params"] = np.log(df["non_embedding_params"])
    df["log_total_params"] = np.log(df["total_params"])
    df["log_flop_matched_non_embedding_params"] = np.log(df["flop_matched_non_embedding_params"])

    # not sure if we should scale n_shot
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['non_embedding_params', 'log_non_embedding_params', 'total_params', 'log_total_params', 'flop_matched_non_embedding_params', 'log_flop_matched_non_embedding_params', 'n_shot']])
    # scaled_df = pd.DataFrame(scaled_data, columns=['non_embedding_params', 'log_non_embedding_params', 'total_params', 'log_total_params', 'flop_matched_non_embedding_params', 'log_flop_matched_non_embedding_params', 'n_shot'])

    X_train = np.concatenate([encoded_data, scaled_data], axis=1)
    df['normalized_score'] = df.apply(normalize_score, axis=1)
    y_train = df["normalized_score"]

    logger.info("Training set size: {}; #categorical: {}; #numeric: {}".format(X_train.shape[0], encoded_data.shape[1], scaled_data.shape[1]))

    if dev_file is not None:
        df = pd.read_csv(dev_file, index_col=False)
        df = df[df["is_preferred_metric"] == 1].reset_index()
        X_dev, y_dev = process_subset(df, encoder, scaler)
        logger.info("Dev set size: {}".format(X_dev.shape[0]))
    else:
        X_dev, y_dev = None, None
        
    if test_file is not None:
        df = pd.read_csv(test_file, index_col=False)
        df = df[df["is_preferred_metric"] == 1].reset_index()
        X_test, y_test = process_subset(df, encoder, scaler)
        logger.info("Test set size: {}".format(X_test.shape[0]))
    else:
        X_test, y_test = None, None

    return X_train, y_train, X_dev, y_dev, X_test, y_test, (encoder, scaler)

def process_subset(df, encoder, scaler):
    encoded_data = encoder.transform(df[['task', 'subtask', 'model_family', 'model_name', 'metric_name']])

    df["log_non_embedding_params"] = np.log(df["non_embedding_params"])
    df["log_total_params"] = np.log(df["total_params"])
    df["log_flop_matched_non_embedding_params"] = np.log(df["flop_matched_non_embedding_params"])

    scaled_data = scaler.transform(df[['non_embedding_params', 'log_non_embedding_params', 'total_params', 'log_total_params', 'flop_matched_non_embedding_params', 'log_flop_matched_non_embedding_params', 'n_shot']])

    X = np.concatenate([encoded_data, scaled_data], axis=1)
    df['normalized_score'] = df.apply(normalize_score, axis=1)
    y = df['normalized_score']

    return X, y
