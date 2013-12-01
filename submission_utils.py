import pandas as pd
import sklearn.metrics
import numpy as np


def write_submission_csv(prediction_df, filename):
    """Take a dataframe of predictions in the same form as data.labels_df and
    write a submission csv"""

    clipnames = sorted(prediction_df.index.values)

    submission = "ID,Probability\n"
    for clipname in clipnames:
        for call_id in sorted(prediction_df.columns.values):
            submission += "%s_classnumber_%s,%f\n" % (clipname, call_id, prediction_df.ix[clipname, call_id])

    with open(filename, 'w') as f:
        f.write(submission)


def parse_submission_csv(filename):

    df = pd.read_csv(filename)
    df['filename'] = df.ID.map(lambda x: x.split('.')[0] + '.wav')
    df['call_id'] = df.ID.map(lambda x: int(x.split('.')[1].split('_')[-1]))
    return df.pivot(index='filename', values='Probability', columns='call_id')


def evaluate(prediction_df, truth_df):

    return sklearn.metrics.roc_auc_score(truth_df.values.ravel(), prediction_df.values.ravel())


def normalized_even_blend(pred_dfs):

    return np.sum([df/df.mean().mean() for df in pred_dfs]) / len(pred_dfs)
