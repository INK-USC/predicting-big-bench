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
