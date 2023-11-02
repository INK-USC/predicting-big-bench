import pandas as pd
import numpy as np
import torch
import os

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
    encoded_data = encoder.fit_transform(df[['task', 'subtask', 'model_family', 'model_name']]) # skip metric name
    print(encoded_data.shape)

    # numerical features
    df["log_non_embedding_params"] = np.log(df["non_embedding_params"])
    df["log_total_params"] = np.log(df["total_params"])
    df["log_flop_matched_non_embedding_params"] = np.log(df["flop_matched_non_embedding_params"])

    # not sure if we should scale n_shot
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['non_embedding_params', 'log_non_embedding_params', 'total_params', 'log_total_params', 'flop_matched_non_embedding_params', 'log_flop_matched_non_embedding_params', 'n_shot']])
    print(scaled_data.shape)

    # new in multi-target: metric name -> indices that will be later used to get the set of regression parameters
    metric_encoder = OrdinalEncoder()
    encoded_metric_name = metric_encoder.fit_transform(df[["metric_name"]])

    X_train = np.concatenate([encoded_data, scaled_data, encoded_metric_name], axis=1) # metric name is the last feature
    df['normalized_score'] = df.apply(normalize_score, axis=1)
    y_train = df["normalized_score"]

    logger.info("Training set size: {}; #categorical: {}; #numeric: {}".format(X_train.shape[0], encoded_data.shape[1], scaled_data.shape[1]))

    if dev_file is not None:
        df = pd.read_csv(dev_file, index_col=False)
        df = df[df["is_preferred_metric"] == 1].reset_index() # train on multiple target but only test on preferred metrics
        X_dev, y_dev = process_subset(df, encoder, scaler, metric_encoder)
        logger.info("Dev set size: {}".format(X_dev.shape[0]))
    else:
        X_dev, y_dev = None, None
        
    if test_file is not None:
        df = pd.read_csv(test_file, index_col=False)
        df = df[df["is_preferred_metric"] == 1].reset_index() # train on multiple target but only test on preferred metrics
        X_test, y_test = process_subset(df, encoder, scaler, metric_encoder)
        logger.info("Test set size: {}".format(X_test.shape[0]))
    else:
        X_test, y_test = None, None

    return X_train, y_train, X_dev, y_dev, X_test, y_test, (encoder, scaler, metric_encoder)

def process_subset(df, encoder, scaler, metric_encoder):
    encoded_data = encoder.transform(df[['task', 'subtask', 'model_family', 'model_name']])

    df["log_non_embedding_params"] = np.log(df["non_embedding_params"])
    df["log_total_params"] = np.log(df["total_params"])
    df["log_flop_matched_non_embedding_params"] = np.log(df["flop_matched_non_embedding_params"])

    scaled_data = scaler.transform(df[['non_embedding_params', 'log_non_embedding_params', 'total_params', 'log_total_params', 'flop_matched_non_embedding_params', 'log_flop_matched_non_embedding_params', 'n_shot']])

    # new in multi-target!
    encoded_metric_name = metric_encoder.transform(df[["metric_name"]])

    X = np.concatenate([encoded_data, scaled_data, encoded_metric_name], axis=1)
    df['normalized_score'] = df.apply(normalize_score, axis=1)
    y = df["normalized_score"]

    return X, y