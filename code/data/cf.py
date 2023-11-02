import pandas as pd

from .utils import normalize_score, METRICS_OF_INTEREST

def preprocess(args, logger, train_file, dev_file, test_file):
    train_df = preprocess_one_file(train_file)
    dev_df = preprocess_one_file(dev_file) if dev_file is not None else None
    test_df = preprocess_one_file(test_file) if test_file is not None else None

    # normally it's X_train, y_train, X_dev, y_dev, X_test, y_test, (encoder, scaler); 
    # for CF methods we are merging x's and y into one df because that's how the surprise package read data
    return train_df, None, dev_df, None, test_df, None, None
    

def preprocess_one_file(filename):
    df = pd.read_csv(filename, index_col=False)

    df = df[df["is_preferred_metric"] == 1].reset_index()

    df = df[df["metric_name"].isin(METRICS_OF_INTEREST)]
    df = df.reset_index()

    df['normalized_score'] = df.apply(normalize_score, axis=1)
    
    df["model_unique"] = df[['model_family', 'model_name']].agg('___'.join, axis=1)
    df["n_shot_str"] = df["n_shot"].astype(str)
    df["task_unique"] = df[['subtask', 'n_shot_str']].agg('___'.join, axis=1)

    new_df = df[["model_unique", "task_unique", "normalized_score"]]
    new_df = new_df.rename(columns={"normalized_score": "score"})

    return new_df