import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, KFold

df = pd.read_csv("../filtered_v2.csv", index_col=False)

grouped_keys = list(df.groupby(["subtask", "model_family"]).groups.keys())
# grouped_keys = np.array(grouped_keys)
print(len(grouped_keys))
# breakpoint()
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(grouped_keys)):
    print(fold_idx)
    start_time = time.time()

    train_keys = [grouped_keys[i] for i in train_idx]
    test_keys = [grouped_keys[i] for i in test_idx]

    test_mask = df.apply(lambda row: (row['subtask'], row["model_family"]) in test_keys, axis=1)

    train_keys, dev_keys = train_test_split(train_keys, test_size=0.1/0.9, random_state=42)

    # train_mask = df.apply(lambda row: (row['subtask'], row["model_family"], row["total_params"], row["n_shot"]) in train_keys, axis=1)
    dev_mask = df.apply(lambda row: (row['subtask'], row["model_family"]) in dev_keys, axis=1)

    train_mask = ~(dev_mask | test_mask)

    train_df = df[train_mask]
    dev_df = df[dev_mask]
    test_df = df[test_mask]

    out_dir = "./{}".format(fold_idx)
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"))
    dev_df.to_csv(os.path.join(out_dir, "dev.csv"))
    test_df.to_csv(os.path.join(out_dir, "test.csv"))

    end_time = time.time()
    print(end_time - start_time)
