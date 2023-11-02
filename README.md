## How Predictable Are Large Language Model Capabilities? A Case Study on BIG-bench

This repository contains code for our paper "How Predictable Are Large Language Model Capabilities? A Case Study on BIG-bench" (EMNLP Findings 2023) [[Preprint]](https://arxiv.org/abs/2305.14947).

### Quick Links
- [Environment](#environment)
- [Data](#data)
- [Training](#training)
- [Searching](#searching) 
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

#### Pre-processing

We've included the pre-processed BIG-bench experiment records in `data/bigbench/`. To run the pre-processing by yourself, see below.

<details>
<summary>Pre-processing</summary>

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
</details>

#### Create different train-test splits
In the paper we defined 5 different ways to create train-test splits, named as `L1/L2.1/L2.2/L3/L4`. 
To create them, go to `data/bigbench/<split_name>` and run the `prep.py` within the folder.

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
