## How Predictable Are Large Language Model Capabilities? A Case Study on BIG-bench

This repository contains code for our paper "How Predictable Are Large Language Model Capabilities? A Case Study on BIG-bench" (EMNLP Findings 2023) [[Preprint]](https://arxiv.org/abs/2305.14947).

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

We've included the pre-processed BIG-bench experiment records in `data/bigbench/`. If you need to rerun the pre-processing by yourself, see below.

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
The whole process will take ~2hrs.

#### Create different train-test splits
In the paper we defined 5 different ways to create train-test splits, named as `L1/L2.1/L2.2/L3/L4`. 
To create them, go to `data/bigbench/<split_name>` and run the `prep.py` within the folder.

### Training Performance Prediction Models

`code/example.sh` contains an example script to reproduce experiments in Sec. 3-4 of the paper.

The script will first automatically tune hyperparameters on the `train_file` and `dev_file`. 
Then it will run the best set of hyperparameters on all folds in the `../data/bigbench/${setting}/` directory.
The predictions will be saved where the data files are. The test set predictions will be saved to `test_mlp_pred.csv` in the following example.


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


### Contact Us

If you have any question, please submit an issue, or reach out to Qinyuan (qinyuany@usc.edu).

If you used our code in your study, or find our paper useful, please cite us use the following bib entry:


```
@article{ye2023predictable,
  title={How Predictable Are Large Language Model Capabilities? A Case Study on BIG-bench},
  author={Ye, Qinyuan and Fu, Harvey Yiyun and Ren, Xiang and Jia, Robin},
  journal={arXiv preprint arXiv:2305.14947},
  year={2023}
}
```
