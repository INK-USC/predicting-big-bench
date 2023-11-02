import random
import os
import pandas as pd

import torch.multiprocessing as mp
from functools import partial

from data.dynamic import preprocess
from models.mlp import run_mlp_regression
from tqdm import tqdm
from .utils import generate_random_combinations
from .fixed import search_run_fixed

def search_beam_search(args, logger, run_func):

    df = pd.read_csv(args.full_file, index_col=False)
    subtasks = list(df["subtask"].unique())

    # candidate_task_lists = generate_random_combinations(subtasks, args.search_budget, args.search_n_trials)

    df = pd.DataFrame(columns=["id", "dev_r2", "n_tasks", "selected_tasks"])
    best_r2, best_task_list = -1e10, None

    beams = [([], 0.0)] # initialize 

    for t in range(args.search_budget):
        visited = set()
        new_beams = []
        new_candidate_task_lists = []

        for beam in beams:

            seq, score = beam

            # get things to search
            if args.search_beam_random < 1.0:
                options = random.sample(subtasks, int(len(subtasks) * args.search_beam_random))
            else:
                options = subtasks

            for candidate_new_task in options:
                new_candidiate_list = seq + [candidate_new_task]
                frozen_set = frozenset(set(new_candidiate_list))
                if frozen_set not in visited:
                    visited.add(frozen_set)
                    new_candidate_task_lists.append(new_candidiate_list)

        for i, task_list in tqdm(enumerate(new_candidate_task_lists)):
            logger.info("Current Search [N_task={}]: {}".format(t+1, task_list))
            dev_r2 = search_run_fixed(args, logger, run_func, task_list)
            # dev_r2 = random.random() # simluate
            df.loc[len(df.index)] = [i, t, dev_r2, task_list]
            new_beams.append((task_list, dev_r2))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:args.search_beam_size]

        if args.save_search_logs:
            df.to_csv(os.path.join(args.output_dir, "search.csv"), index=False)

    best_task_list, best_r2 = beams[0]
    return best_r2, best_task_list, df

