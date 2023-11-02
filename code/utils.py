import torch
import random
import numpy as np
import pandas as pd

from data.utils import normalize_score

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_predictions(filename, predictions, identifier="null"):
    # Note: for now we don't need to transform back becasue the preferred metrics are always in [0.0, 1.0]
    df = pd.read_csv(filename, index_col=False)
    df = df[df["is_preferred_metric"] == 1].reset_index() # for dev/test cases we only care about preferred metric
    df['normalized_score'] = df.apply(normalize_score, axis=1)
    df["predicted_score"] = predictions 
    df["diff"] = np.abs(df["normalized_score"] - predictions)

    new_filename = filename.replace(".csv", "_{}_pred.csv".format(identifier))
    # i don't even know how they showed up...
    df.drop('index', axis=1, inplace=True)
    df.drop('Unnamed: 0', axis=1, inplace=True)

    df.to_csv(new_filename)
