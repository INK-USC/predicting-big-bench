import pandas as pd

df = pd.read_csv("../data/bigbench/all.csv", index_col=False)
print("# all entries: ", len(df))

## STEP 1: keeping entries using "BIG-G sparse", "BIG-G T=0", "BIG-G T=1", "PaLM", "GPT", "Gopher" models;
##         exclude T0/T5 models
df1 = df[df["model_family"].isin(["BIG-G sparse", "BIG-G T=0", "BIG-G T=1", "PaLM", "GPT", "Gopher"])]
print("step 1, keep entries using BIG-G/PaLM/GPT/Gopher model: ", len(df1))

## STEP 2: delete tasks whose performance (on preferred metric) is all zero
df1["task_and_subtask_name"] = df1[['task', 'subtask']].agg('___'.join, axis=1)
df11 = df1[df1["is_preferred_metric"] == 1] # keep only preferred metric rows
zero_score_tasks = df11.groupby('task_and_subtask_name').filter(lambda x: (x['score'] == 0).all())
all0subtasks = zero_score_tasks['task_and_subtask_name'].unique()
print("tasks with all 0 performance on preferred metric: {}".format(all0subtasks))

df2 = df1[~df1["task_and_subtask_name"].isin(all0subtasks)]
print("step 2, remove subtasks with all 0 performance: ", len(df2))
df2 = df2.drop(columns=["task_and_subtask_name"])

## STEP 3
# keep experiments whose preferred metric is in ["exact_str_match", "multiple_choice_grade", "rougeLsum"]
df3 = df2[df2["is_preferred_metric"] == 1]
df3 = df3[df3["metric_name"].isin(["exact_str_match", "multiple_choice_grade", "rougeLsum"])]

# keep experiments and keep their performance with other metrics (if avaialble in df2)
tasks_using_preferred_metrics = df3["subtask"].unique()
df4 = df2[df2["subtask"].isin(tasks_using_preferred_metrics)]
print("step 3, remove experiments whose perferred metric is not in (exact_str_match, multiple_choice_grade, rougeLsum): ", len(df4))

## STEP 4
# remove experiments whose scores are aggregated from sub-tasks 
mask = (df4["task"] == df4["subtask"]) & (df4.groupby("task")["subtask"].transform("nunique") > 1)
df4 = df4[~mask]
df3 = df4[df4["is_preferred_metric"] == 1]
#  df[mask]["subtask"].unique() # to see which tasks have these special entries

df4.to_csv("../data/bigbench/filtered.csv", index=False)
print("====\n#entried with all metrics: {}".format(len(df4)))

df3.to_csv("../data/bigbench/filtered_preferred_only.csv", index=False)
print("#entried with preferred metric: {}".format(len(df3)))

## STEP 5
# remove subtasks with <100 examples (b/c small sample size -> large variance -> unpredictable)
all_subtasks = df4["subtask"].unique()
to_keep = []
for subtask in all_subtasks:
    normalized_name = subtask.replace(":", "___")
    text_file = "../data/bigbench_raw_text/{}_all.jsonl".format(normalized_name)
    with open(text_file) as fin:
        lines = fin.readlines()
    if len(lines) > 100:
        to_keep.append(subtask)
    else:
        print("excluding {}: size {}".format(subtask, len(lines)))

df4 = df4[df4["subtask"].isin(to_keep)]
df3 = df4[df4["is_preferred_metric"] == 1]

df4.to_csv("../data/bigbench/filtered_v2.csv", index=False)
print("====\n#entried with all metrics: {}".format(len(df4)))

df3.to_csv("../data/bigbench/filtered_preferred_only_v2.csv", index=False)
print("#entried with preferred metric: {}".format(len(df3)))
