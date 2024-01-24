## How Predictable Are Large Language Model Capabilities? A Case Study on BIG-bench

This repository contains code for our paper "How Predictable Are Large Language Model Capabilities? A Case Study on BIG-bench" (EMNLP Findings 2023). [[Paper]](https://aclanthology.org/2023.findings-emnlp.503/) [[Video]](https://youtu.be/cycBv5Pbn50) [[Slides]](https://yeqy.xyz/src/BIG-bench-analysis/BIG-bench-analysis-slides.pdf) [[Poster]](https://yeqy.xyz/src/BIG-bench-analysis/BIG-bench-analysis-poster.pdf) [[Tweet]](https://x.com/qinyuan_ye/status/1722682948066193441?s=20)

### Quick Links
- [Environment](#environment)
- [Data](#data)
- [Training Performance Prediction Models](#training-performance-prediction-models)
- [Searching for "small-bench"](#searching-for-small-bench) 
- [Contact Us](#contact-us)

### Environment
```bash
conda create --name pbb python=3.9
conda activate pbb
conda install cudatoolkit=11.3 -c anaconda
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
pip install -r requirements.txt
```

### Data

#### Pre-processing (Optional)

We've included the pre-processed BIG-bench experiment records in `data/bigbench/`. If you need to rerun the pre-processing by yourself, please use the script below. The whole process will take ~2hrs.


```bash
# clone BIG-bench in a separate folder
cd ..
git clone https://github.com/google/BIG-bench.git
# come back to explogs directory and run the script
cd predicting-big-bench/data_prep
# gather experiment logs from BIG-Bench directory
python big_bench.py
# filter the logs to formulat the dataset
python filter_big_bench.py
```

#### Create different train-test splits
In the paper we defined 5 different ways to create train-test splits, named as `L1/L2.1/L2.2/L3/L4`. 
To create them, go to `data/bigbench/<split_name>` and run the `prep.py` within the folder.

### Training Performance Prediction Models

`code/scripts/train.sh` contains an example script to reproduce experiments in Sec. 3-4 of the paper.

* The script will first automatically tune hyperparameters on the `train_file` and `dev_file`. 
* Then it will run the best set of hyperparameters on all folds in the `../data/bigbench/${setting}/` directory.
* The mean and std over all folds will be printed out at the end of the program.
* The predictions will be saved where the data files are. For example, the test set predictions from a MLP model will be saved to `test_mlp_pred.csv`.


```bash
setting="l1" # select from "l1", "l2_1", "l2_2", "l3", "l4"
model_arch="mlp" # select from "bsl_model_task", "svd", "task_task_knn", "model_model_knn","random_forest", "xgb", "mlp"
n_trials=200 # random hyperparameter combinations to try

python cli.py \
--train_file ../data/bigbench/${setting}/0/train.csv \
--dev_file ../data/bigbench/${setting}/0/dev.csv \
--test_file ../data/bigbench/${setting}/0/test.csv \
--data_dir ../data/bigbench/${setting}/ \
--mode tunehp_then_multi_run \
--model_arch ${model_arch} \
--output_dir output/${setting}/${model_arch} \
--preferred \
--save_predictions \
--n_trials ${n_trials}
```


### Searching for "small-bench"

#### Step 1: Search
* `code/scripts/search_random5000.sh` contains the script to reproduce "Best of 5000" described in the paper.
* `code/scripts/search_greedy.sh` contains the script to reproduce greedy search described in the paper.
* The code supports more search methods such as beam search, simulated annealing, etc. You can specify this in `--search_mode`. Also please make sure to check `cli.py` for args specific to a method.

#### Step 2: Post-process
* We include the post-processing scripts in `data/smallbench`. They are named as `prep.py`
* The post-processing results are saved to a csv file. We have included search results of Best of 5000, Greedy Search, K-means, K-means + Task Value in `data/smallbench`.
* For example, `data/smallbench/random5000/random5000.csv` is the search results for the "Best of 5000" method.

#### Step 3: Eval
* `code/scripts/search_eval_bbhard_and_bblite.sh` contains the scripts to evaluate the predictions when BIG-bench Hard / Lite are used as the "small-bench" to recover performance on remaining tasks.
* `code/scripts/search_eval_greedy.sh` contains the scripts to evaluate the search results of greedy search. By changing the `--selected_tasks` args you can evaluate search results of other methods by pointing to the corresponding csv file.
* Evaluation results can be found in the `--output_dir` that is specified in the script. The file that ends with `_summary.csv` contains the mean and std of 30-fold cross validation.

### Contact Us

If you have any question, please submit an issue, or reach out to Qinyuan (qinyuany@usc.edu).

If you used our code in your study, or find our paper useful, please cite us with the bibkey `ye-etal-2023-predictable` in the official ACL Anthology, or use the following BibTeX:

<details>
<summary>BibTeX</summary>

```
@inproceedings{ye-etal-2023-predictable,
    title = "How Predictable Are Large Language Model Capabilities? A Case Study on {BIG}-bench",
    author = "Ye, Qinyuan  and
      Fu, Harvey  and
      Ren, Xiang  and
      Jia, Robin",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.503",
    doi = "10.18653/v1/2023.findings-emnlp.503",
    pages = "7493--7517",
}
```
</details>