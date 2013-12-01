import numpy as np
import pandas as pd

import copy
import random
from collections import namedtuple

import data


Fold = namedtuple('Fold', 'training_mask, testing_mask, clipnames, call_ids, label_df')


class FoldBuilder(object):
    """Build folds for submission or cross validation"""

    def __init__(self):

        training_wav_dict = data.get_wav_dict(test=False)
        testing_wav_dict = data.get_wav_dict(test=True)

        self.wav_dict = {}
        self.wav_dict.update(training_wav_dict)
        self.wav_dict.update(testing_wav_dict)

        self.training_clipnames = sorted(training_wav_dict.keys())
        self.testing_clipnames = sorted(testing_wav_dict.keys())

        training_wavs = [self.wav_dict[clipname] for clipname in self.training_clipnames]
        testing_wavs = [self.wav_dict[clipname] for clipname in self.testing_clipnames]

        self.clipnames = np.array(self.training_clipnames + self.testing_clipnames)

        self.list_of_wavs = training_wavs + testing_wavs
        self.clipname_to_index = dict((clipname, i) for i, clipname in enumerate(self.clipnames))


    def get_wavs(self):
        return self.list_of_wavs


    def _build_mask(self, clipnames):

        mask = np.repeat(False, len(self.list_of_wavs))
        for name in clipnames:
            mask[self.clipname_to_index[name]] = True
        return mask


    def make_fold(self, training_clipnames, testing_clipnames):

        training_mask = self._build_mask(training_clipnames)
        testing_mask = self._build_mask(testing_clipnames)
        return Fold(training_mask, testing_mask,  self.clipnames, range(1, 88), data.label_df)


    def submission_fold(self):

        return self.make_fold(self.training_clipnames, self.testing_clipnames)


    def make_cvd_folds(self, number_folds, training_proportion, testing_proportion, shuffle=True, random_seed=None):

        train_count = int(training_proportion*len(self.training_clipnames))
        test_count = int(testing_proportion*len(self.training_clipnames))

        if random_seed != None:
            random.seed(random_seed)

        folds = []
        for _ in range(number_folds):
            clipnames = copy.deepcopy(self.training_clipnames)
            if shuffle:
                random.shuffle(clipnames)
            training_clipnames = clipnames[:train_count]
            testing_clipnames = clipnames[-test_count:]
            folds.append(self.make_fold(training_clipnames, testing_clipnames))

        return folds

