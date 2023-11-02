import random
import os
import pandas as pd
import numpy as np

from data.dynamic import preprocess
from models.mlp import run_mlp_regression
from tqdm import tqdm, trange

from .utils import generate_random_combinations
from .fixed import search_run_fixed
from .genetic_algorithm import mutation

N_TRIALS = 5
TOTAL_EPOCHS = 1000

INIT_TEMP = 100
ALPHA = 0.99

def search_simulated_annealing(args, logger, run_func):

    df = pd.read_csv(args.full_file, index_col=False)
    subtasks = list(df["subtask"].unique())

    # generate initial population
    init_selections = generate_random_combinations(subtasks, args.search_budget, N_TRIALS)

    df = pd.DataFrame(columns=["id", "timestamp", "dev_r2", "n_tasks", "selected_tasks"])
    best_r2, best_task_list = -1e10, None

    for i in range(N_TRIALS):
        temperature = INIT_TEMP
        selection = init_selections[i]

        # score = random.random() # to check the overall workflow
        score = search_run_fixed(args, logger, run_func, selection)
        df.loc[len(df.index)] = [i, 0, score, args.search_budget, selection]
        
        if score > best_r2:
            best_r2 = score
            best_task_list = selection

        for j in range(TOTAL_EPOCHS):

            new_selection = mutation(selection, subtasks) # me being lazy
            # new_score = random.random() # to check the overall workflow
            new_score = search_run_fixed(args, logger, run_func, new_selection)
            df.loc[len(df.index)] = [i, j+1, new_score, args.search_budget, new_selection]

            if score > best_r2:
                best_r2 = new_score
                best_task_list = new_selection

            delta_e = new_score - score

            if (delta_e > 0) or (np.exp(delta_e / temperature) > random.random()):
                selection = new_selection
                score = new_score

            temperature = temperature * ALPHA
        
        # save after each trial
        df.to_csv(os.path.join(args.output_dir, "search.csv"), index=False)

    return best_r2, best_task_list, df
