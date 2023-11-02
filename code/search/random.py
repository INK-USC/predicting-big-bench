import pandas as pd

import torch.multiprocessing as mp
from functools import partial

from data.dynamic import preprocess
from models.mlp import run_mlp_regression
from tqdm import tqdm
from .utils import generate_random_combinations
from .fixed import search_run_fixed

def search_task_list(args, logger, run_func, task_list):
    # wrapper for easier mp
    return (task_list, search_run_fixed(args, logger, run_func, task_list))

def search_random(args, logger, run_func):

    df = pd.read_csv(args.full_file, index_col=False)
    subtasks = list(df["subtask"].unique())

    candidate_task_lists = generate_random_combinations(subtasks, args.search_budget, args.search_n_trials)

    df = pd.DataFrame(columns=["id", "dev_r2", "selected_tasks"])
    best_r2, best_task_list = -1e10, None

    if args.search_n_jobs > 1:
        pool = mp.Pool(processes=args.search_n_jobs)

        # for i, task_list in enumerate(candidate_task_lists):
        #     results.append(pool.apply_async(search_task_list. args=(args, logger, run_func, task_list)))
        search_func = partial(search_task_list, args, logger, run_func)
        results = pool.imap(search_func, candidate_task_lists)

        for i, result in enumerate(results):
            task_list, dev_r2 = result
            df.loc[len(df.index)] = [i, dev_r2, task_list]

            if dev_r2 > best_r2:
                best_r2 = dev_r2
                best_task_list = task_list
    else:

        for i, task_list in tqdm(enumerate(candidate_task_lists)):
    
            dev_r2 = search_run_fixed(args, logger, run_func, task_list)
            df.loc[len(df.index)] = [i, dev_r2, task_list]

            logger.info("Run {}, Dev R2 = {}".format(i, dev_r2))

            if dev_r2 > best_r2:
                best_r2 = dev_r2
                best_task_list = task_list
        
        logger.info("Best Dev R2: {}".format(best_r2))
        logger.info("Best list of tasks: {}".format(best_task_list))
    
    return best_r2, best_task_list, df
