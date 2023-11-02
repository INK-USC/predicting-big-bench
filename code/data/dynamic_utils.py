import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from .utils import normalize_score

def prepare_dataset_from_df(logger, train_df, dev_df, test_df):

    # categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_data = encoder.fit_transform(train_df[['task', 'subtask', 'model_family', 'model_name', 'metric_name']])

    # numerical features
    train_df["log_non_embedding_params"] = np.log(train_df["non_embedding_params"])
    train_df["log_total_params"] = np.log(train_df["total_params"])
    train_df["log_flop_matched_non_embedding_params"] = np.log(train_df["flop_matched_non_embedding_params"])

    # not sure if we should scale n_shot
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(train_df[['non_embedding_params', 'log_non_embedding_params', 'total_params', 'log_total_params', 'flop_matched_non_embedding_params', 'log_flop_matched_non_embedding_params', 'n_shot']])
    # scaled_df = pd.DataFrame(scaled_data, columns=['non_embedding_params', 'log_non_embedding_params', 'total_params', 'log_total_params', 'flop_matched_non_embedding_params', 'log_flop_matched_non_embedding_params', 'n_shot'])

    X_train = np.concatenate([encoded_data, scaled_data], axis=1)
    train_df['normalized_score'] = train_df.apply(normalize_score, axis=1)
    y_train = train_df["normalized_score"]

    logger.info("Training set size: {}; #categorical: {}; #numeric: {}".format(X_train.shape[0], encoded_data.shape[1], scaled_data.shape[1]))

    if dev_df is not None and len(dev_df) > 0:
        X_dev, y_dev = process_subset(dev_df, encoder, scaler)
        logger.info("Dev set size: {}".format(X_dev.shape[0]))
    else:
        X_dev, y_dev = None, None

    if test_df is not None and len(test_df) > 0:
        X_test, y_test = process_subset(test_df, encoder, scaler)
        logger.info("Test set size: {}".format(X_test.shape[0]))
    else:
        X_test, y_test = None, None

    return X_train, y_train, X_dev, y_dev, X_test, y_test, (encoder, scaler, train_df, dev_df, test_df)

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

def prepare_dataset_from_df_for_cf(logger, train_df, dev_df, test_df):
    # a different process for collaborative filtering / recommendation methods
    train_df = preprocess_one_df_for_cf(train_df)
    dev_df = preprocess_one_df_for_cf(dev_df) if dev_df is not None else None
    test_df = preprocess_one_df_for_cf(test_df) if test_df is not None else None

    # normally it's X_train, y_train, X_dev, y_dev, X_test, y_test, (encoder, scaler); 
    # for CF methods we are merging x's and y into one df because that's how the surprise package read data
    return train_df, None, dev_df, None, test_df, None, None
    

def preprocess_one_df_for_cf(df):
    df['normalized_score'] = df.apply(normalize_score, axis=1)
    
    df["model_unique"] = df[['model_family', 'model_name']].agg('___'.join, axis=1)
    df["n_shot_str"] = df["n_shot"].astype(str)
    df["task_unique"] = df[['subtask', 'n_shot_str']].agg('___'.join, axis=1)

    new_df = df[["model_unique", "task_unique", "normalized_score"]]
    new_df = new_df.rename(columns={"normalized_score": "score"})

    return new_df