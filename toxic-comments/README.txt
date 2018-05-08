This readme will explain the file structure and explain the python scripts

A link to my kaggle profile, proving the competition was entered and submissions were made:
https://www.kaggle.com/jfc14cmu

STRUCTURE:
the python scripts within the same directory as the read me relate to the kaggle competition "Identifying Toxic Comments"

eda.py - The script used for exploratory data analysis

ensemble.py - the script used for parameter tuning and ensembling experiments

extraction_experiments.py - the script used for parameter turning and feature extraction experiments

fs_experiments.py - the script used for feature selection parameter tuning and experiments

model_experiments.py - the script used for model parameter tuning and experiments

stacking_experiment.py - the script used for feature stacking experiments

toxic_comments - the script that contains all functions that gather results used in experiments within the dissertation, 
		also contains functions that made csv files for competition submission

benchmarks.csv - Contains the parameter tuning results, may not contain all values tested.

DIRECTORY - submissions - contains some of the submissions used in the kaggle competition.

DIRECTORY - benchmarking - contains all benchmark datasets and python scripts used to gather results used in dissertation
Each script is named similarly to the dataset for ease of use.
Each script contains the experiments performed on a given dataset, experiments can be found under a comment showing the experiment name
For example - #Feature Extraction - would show the section of code that performed the feature extraction experiments
Most experiments are commented out - To test them simply uncomment the section of code
Most experiments make use of the same dictionary 'd', it is important to only uncomment one experiment at a time when running them
It should be noted that not all scripts performed all 5 experiments, some experiments were not suitable for a given dataset


Below is a name of all benchmark dataset scripts
amazon_reviews.py
author_ident.py
cancer.py
movie_reviews.py
movie_reviews2.py
q_pairs.py
search_rel.py
titanic.py
wine.p

SUBDIRECTORY - critdiff
This subdirectory contains the code used to generate the critical difference diagrams, it also contains the csv files used to generate them
critdiff.m - Provided by Tony Bagnall in the machine learning module
drawCDExample.m - used to produce Critical Difference diagrams.
