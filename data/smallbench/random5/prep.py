import random
import pandas as pd

def generate_random_combinations(task_lists, budget, n_trials, seed=42):
    random.seed(seed)

    samples = set()
    task_list = []

    while len(samples) < n_trials:
        sample = sorted(random.sample(task_lists, budget))
        if tuple(sample) not in samples:
            samples.add(tuple(sample))
            task_list.append(sample)

    return task_list

full_file = "../bigbench/filtered_v2.csv"
df = pd.read_csv(full_file, index_col=False)
subtasks = list(df["subtask"].unique())
for budget in [4,8,16,24,32,42]:
    candidate_task_lists = generate_random_combinations(subtasks, budget, 5)
    filename = "random5_ntasks{}.csv".format(budget)
    df = pd.DataFrame({"task_list": candidate_task_lists})
    df.to_csv(filename, index=False)