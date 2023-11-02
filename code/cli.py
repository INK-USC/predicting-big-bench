import os
import sys
import argparse
import logging

import pandas as pd

from run import run
from utils import seed_everything

def parse_args():
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument("--mode", type=str, default="run", choices=["run", "tunehp", "multi_run", "tunehp_then_multi_run", "search"])
    parser.add_argument("--n_trials", type=int, default=100, help="how many times to run random hyperparameter search")
    parser.add_argument("--data_mode", type=str, default="basic", choices=["basic", "multitarget", "external_feature", "cf", "no_model_name"])
    parser.add_argument("--hp", type=str, default=None,
        help="pass in method specific hyperparameters (usually determined by tuning, and pass in for k-fold CV)")

    # experiment specification
    parser.add_argument("--output_dir", default="./output/tmp", type=str)
    parser.add_argument("--save_predictions", action='store_true')
    parser.add_argument("--save_tuning_logs", action="store_true")
    parser.add_argument("--controlled_test", action="store_true")

    # data
    parser.add_argument("--train_file", type=str, default="../data/bigbench/all_metrics_toy/train.csv")
    parser.add_argument("--dev_file", type=str, default="../data/bigbench/all_metrics_toy/dev.csv")
    parser.add_argument("--test_file", type=str, default="../data/bigbench/all_metrics_toy/test.csv")
    parser.add_argument("--full_file", type=str, default="../data/bigbench/all_metrics_toy/full.csv",
        help="only used when --mode==search")
    parser.add_argument("--data_dir", type=str, default="../data/bigbench/all_metrics_toy/", 
        help="in multi_run mode, will read all sub directories in these directory and run multiple times")
    parser.add_argument("--use_external_features", action='store_true')
    parser.add_argument("--external_features_dir", type=str, default="../data/bigbench_bitfit_emb")
    parser.add_argument("--preferred", action='store_true',
        help="keeping only preferred metric entries in the training set")
        # note: dev/test will automatically do this

    # model
    parser.add_argument("--model_arch", type=str, default="MLP", 
        choices=["mlp", "multitarget_mlp", 
                "random_forest", "xgb", 
                "svd", "svdpp", "random", "bsl_model", "bsl_task", "bsl_model_task",
                "task_task_knn", "model_model_knn",
            ])
    parser.add_argument("--hidden_dims", type=str, default="(128,64,32)")
    parser.add_argument("--dropout", type=float, default=0.0)

    # search
    parser.add_argument("--search_mode", type=str, default="fixed", 
        choices=["fixed", "random", "greedy", "beam", "factored_beam", "ga", "sa"]
    )
    parser.add_argument("--selected_tasks", type=str, default=None)
    parser.add_argument("--search_n_trials", type=int, default=100, help="how many times to to run random search")
    parser.add_argument("--search_budget", type=int, default=24, help="target number of tasks")
    parser.add_argument("--search_subroutine", type=str, default="mlp")
    parser.add_argument("--search_beam_size", type=int, default=4)
    parser.add_argument("--search_beam_random", type=float, default=0.25)
    parser.add_argument("--search_cv_level", type=str, default="l1", choices=["l0", "l1", "l2"])
    parser.add_argument("--search_n_jobs", type=int, default=1, help="by default (1), don't use multiprocess")
    parser.add_argument("--save_search_logs", action="store_true")

    # optim
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # others
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action='store_true', 
        help="whether use tqdm or print out intermediate logs")
    parser.add_argument("--rebuttal", action='store_true', 
        help="special flag for rebuttal experiments")

    args = parser.parse_args()
    args.hidden_dims = eval(args.hidden_dims)

    return args

def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                        logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    run(args, logger)

if __name__=='__main__':
    main()