# deep-learning_project-1

Checks:
- data directory has to follow the structure of CIFAR-10 from [Kaggle](https://www.kaggle.com/competitions/cifar-10/overview), both the `train.7z` and `test.7z` should be unzipped

# Creating a conda environment

To create proper conda environment, use

`conda create --name myenv --file conda_requirements.yml`

in root directory of this repository.

# Running experiments

To run an experiment, use

`python -m experiments/_X/exp.py`

where X - number of the experiment.