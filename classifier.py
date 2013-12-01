import numpy as np
import pandas as pd

import sklearn
import sklearn.ensemble


class LabelDistributingBinaryForest:
    """
    Wraps the multi-output random forest in sklearn to use folds. Labels for each wav
    are distributed to each window in that wav.
    """

    def __init__(self, forest_params={}):
        self.set_params(forest_params=forest_params)
        self.clf = sklearn.ensemble.RandomForestClassifier(**forest_params)

    def set_params(self, **params):
        self.params = params

    def get_params(self):
        return self.params

    def fit(self, list_of_features, fold):

        # Use training_mask to limit the inputs nad outputs to training examples
        label_matrix = fold.label_df.ix[fold.clipnames[fold.training_mask]].values
        feature_array_list = [features for is_training, features in zip(fold.training_mask, list_of_features) if is_training]

        label_array_list = []

        for i, feature_array in enumerate(feature_array_list):
            label_array = label_matrix[i]
            # Repeat the labels for each window of features
            expanded_label_array = np.repeat(np.atleast_2d(label_array), feature_array.shape[0], axis=0)
            label_array_list.append(expanded_label_array)

        feature_matrix = np.vstack(feature_array_list)
        label_matrix = np.vstack(label_array_list)

        self.clf.fit(feature_matrix, label_matrix)

    def transform(self, list_of_features):

        feature_matrix = np.vstack(list_of_features)
        splits = np.cumsum([w.shape[0] for w in list_of_features])[:-1]
        # If the random forest hasn't seen both a positive and negative case of a particular label,
        # it returns a column filled with [1] instead of [0.2 0.8].
        probs_weird_structure = self.clf.predict_proba(feature_matrix)
        # We pick out the probs for the label not being present because it is more common for a positive
        # case to be missing than a negative case
        # TODO: make this handle the case when the negative label is missing
        indices_of_negative_class = [np.where(class_list==0)[0][0] for class_list in self.clf.classes_]
        # Here we pull out the probability for the negative case from what will usually be lists of [p 1-p]
        probs_of_negative_case = [prob[:,index] for prob, index in zip(probs_weird_structure, indices_of_negative_class)]
        probs_of_positive_case = 1-np.array(probs_of_negative_case).T

        return np.split(np.array(probs_of_positive_case), splits, axis=0)
