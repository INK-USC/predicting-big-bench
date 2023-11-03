import os
import pandas as pd

RUN_LOGS_DIR="../../../code/output/search/random5000"

all_lists = []
for n_tasks in [4,8,16,24,32,42]:
# for n_tasks in [42]:
    filename = os.path.join(RUN_LOGS_DIR, "ntask{}".format(n_tasks), "search.csv")
    df = pd.read_csv(filename, index_col=None)
    max_idx = df['dev_r2'].idxmax()

    all_lists.append(df.iloc[max_idx]["selected_tasks"])
    print(df.iloc[max_idx]["dev_r2"])

filename = "random5000.csv"
df = pd.DataFrame({"task_list": all_lists})
df.to_csv(filename, index=False)