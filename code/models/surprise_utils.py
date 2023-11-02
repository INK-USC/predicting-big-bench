import numpy as np
import pandas as pd
from surprise import accuracy

class CustomFold():
    # a hack to use custom train and dev set in sup
    def split(self, data):
        self.n_splits = 1
        trainset = data[0].build_full_trainset()
        testset = data[1].build_full_trainset().build_testset()
        yield trainset, testset

    def get_n_folds(self):
        return 1

def surprise_eval(predictions):
    # get rmse, r2, raw_predictions based on the surprise prediction format
    
    mse = accuracy.mse(predictions, verbose=True)
    rmse = np.sqrt(mse)

    y_mean = np.mean([true_r for (_, _, true_r, est, _) in predictions])
    base = np.mean([float((true_r - y_mean) ** 2) for (_, _, true_r, est, _) in predictions])
    r2 = 1 - mse / base

    raw_predictions = np.array([est for (_, _, true_r, est, _) in predictions])

    return rmse, r2, raw_predictions

def reorder_surprise_preds(df, predictions):
    new_df = pd.DataFrame(columns=["model_unique", "task_unique", "predicted_score"])
    score_dict = {}
    for item in predictions:
        model, task, _, y_pred, _ = item
        score_dict[(model, task)] = y_pred
        
    df['predicted_score'] = [score_dict.get((model, task)) for model, task in zip(df['model_unique'], df['task_unique'])]
    return df['predicted_score'].to_numpy()
