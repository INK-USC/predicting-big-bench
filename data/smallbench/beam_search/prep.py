import os
import pandas as pd

RUN_LOGS_DIR="../../../code/output/search/beam"

all_lists = []

filename = os.path.join(RUN_LOGS_DIR, "search.csv")
df = pd.read_csv(filename, index_col=None)

# weird column name problem
df = df.rename(columns={"n_tasks": "temp"})
df = df.rename(columns={"dev_r2": "n_tasks"})
df = df.rename(columns={"temp": "dev_r2"})
df["n_tasks"] = df["n_tasks"] + 1
print(df.head())
max_scores = df.groupby('n_tasks').apply(lambda x: x.loc[x['dev_r2'].idxmax()])
print(max_scores)

for n_tasks in [4, 8, 16, 24, 32, 42]:
    
    row = max_scores.loc[n_tasks]
    all_lists.append(row["selected_tasks"])
    print("({}, {})".format(n_tasks, row["dev_r2"]))

filename = "beam_search.csv"
df = pd.DataFrame({"task_list": all_lists})
df.to_csv(filename, index=False)