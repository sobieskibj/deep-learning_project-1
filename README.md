# Deep learning course - project I

This repository contains code needed to reproduce the results for project I of the Deep Learning course.

# Getting the dataset

To be able to run the experiments:
- download the CIFAR-10 dataset from [Kaggle](https://www.kaggle.com/competitions/cifar-10/overview)
- place the downloaded directory in the root directory of this repository
- unpack `train.7z` and `test.7z` inside it

# Creating a conda environment

To create proper conda environment, use

`conda create --name myenv --file conda_requirements.yml`

or

`conda create --name myenv --file conda_requirements.txt`

in root directory of this repository.

# Running experiments

To run an experiment, use

`python -m experiments/_X/exp.py`

where X - number of the experiment.