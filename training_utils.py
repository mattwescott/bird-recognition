import copy
import itertools
import random
import os
import joblib
from collections import namedtuple

import numpy as np
import pandas as pd


import config


def combine_configs(**configs):

    config_tuples = list(itertools.product(*configs.values()))

    master_config_list = []

    for config_tuple in config_tuples:
        master_config_list.append(
            dict(zip(configs.keys(), config_tuple))
        )

    return master_config_list


Model = namedtuple('Model', 'run_name, index_within_run, config, clf, oob_score, clf_output_fold, pred_df')


def dump_model(model):

    dump_path = config.OUTPUT_DIR
    relative_path = 'models/%s/%s' % (model.run_name, model.index_within_run)
    directory = os.path.join(dump_path, relative_path)
    os.makedirs(directory)

    joblib.dump(model, os.path.join(directory, 'root'))


def load_model(run_name, index_within_run):

    dump_path = config.OUTPUT_DIR
    relative_path = 'models/%s/%s' % (run_name, index_within_run)
    directory = os.path.join(dump_path, relative_path)

    return joblib.load(os.path.join(directory, 'root'))


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)

