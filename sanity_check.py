import sys

import numpy as np
import pandas as pd

import sklearn.metrics as met


def parse_submission_csv(filename):

    df = pd.read_csv(filename)
    df['filename'] = df.ID.map(lambda x: x.split('.')[0] + '.wav')
    df['call_id'] = df.ID.map(lambda x: int(x.split('.')[1].split('_')[-1]))
    return df.pivot(index='filename', values='Probability', columns='call_id')


if __name__ == '__main__':

    dfs = []
    first = True
    for f in sys.argv:
        if first:
            first=False
            continue
        dfs.append(parse_submission_csv(f))

    df1, avg1 = dfs[0], np.sort(dfs[0].values.ravel())[87000*.975]
    df2, avg2 = dfs[1], np.sort(dfs[1].values.ravel())[87000*.975]

    v1 = met.roc_auc_score(np.greater(df1.values.ravel(), np.ones_like(df1.values.ravel())*avg1), df2.values.ravel())

    v2 = met.roc_auc_score(np.greater(df2.values.ravel(), np.ones_like(df1.values.ravel())*avg2), df1.values.ravel())

    print v1, v2

    if (v1+v2)/2 > 0.96:
        print "Submission passes sanity check"
    else:
        print "Submission fails sanity check"
