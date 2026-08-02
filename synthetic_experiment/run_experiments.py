"""
Run experiment where we train transformer models to learn
De Bruijn DAG structures.


A decoder-only transformer is trained from scratch on paths sampled uniformly
from D_pi = F[B(V,n); pi, N, S]. There is no prompt: the model has to learn
which subset of the vocabulary is legal at each point, not which branch is
correct (that is experiment 1b's job). 

Figures
-------
samples_vs_states     samples to reach the illegal-mass
                        criterion vs transition count, with the
                        coverage-only (no learning) reference.
"""


import os
import sys

import numpy as np
from matplotlib.lines import Line2D

from generate_data import build_dag, ReasoningGenerator
from train_eval import Task, TrainConfig, train, evaluate, training_statistics, coverage_curve
from runner import CACHE_DIR, FIG_DIR, run_sweep, load_cache, rows, crossing_point, save_cache
import plotting as P



