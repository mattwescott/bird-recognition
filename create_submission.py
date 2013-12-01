import pandas as pd
import numpy as np

import transform
import classifier
import FoldBuilder
import training_utils
import submission_utils


def build_model(run_name, index_within_run, config, list_of_wavs, fold):

    training_utils.set_seeds(config['rng_config']['random_seed'])


    def build_features(list_of_wavs, fold):

        mfcc_generator = transform.GenerateMFCCs(**config['mfcc_config'])
        window_mfccs = transform.WindowMFCCs(**config['window_config'])
        window_summarizer = transform.PCASummarizeWindowedMFCCs()

        mfccs = mfcc_generator.transform(list_of_wavs)
        windowed_mfccs = window_mfccs.transform(mfccs)
        summarized_mfccs = window_summarizer.transform(windowed_mfccs)

        return summarized_mfccs

    clf = classifier.LabelDistributingBinaryForest(forest_params=config['clf_config']['forest_params'])

    def make_predictions(clf_output, fold):

        prediction_collapser = transform.CollapseWindowPredictions()

        list_of_preds = prediction_collapser.transform(clf_output)

        return transform.make_prediction_df(list_of_preds, fold)

    features = build_features(list_of_wavs, fold)
    clf.fit(features, fold)
    clf_output = clf.transform(features)
    pred_df = make_predictions(clf_output, fold)

    return training_utils.Model(run_name, index_within_run, config, None,
                                clf.clf.oob_score_, clf_output, pred_df)


if __name__ == '__main__':


    mfcc_config = [
    
        dict(
            numcep = 14,
            nfilt= 21,
            lowfreq=500,
            highfreq=None,
            cornerfreq=1500.
        ),
    
    ]
    
    window_config = [
    
        dict(
            window_length = 149,
            overlap = 0.9,
        )
    
    ]
    
    rng_config = [dict(random_seed=i) for i in range(40, 40+16)]
    
    clf_config = [
    
        dict(
            forest_params = dict(
                                n_estimators = 100,
                                n_jobs = 6,
                                min_samples_leaf = 2,
                                min_samples_split = 5,
                                oob_score=True,
                            )
        )
    
    ]

    configs = training_utils.combine_configs(
        mfcc_config=mfcc_config,
        window_config=window_config,
        clf_config=clf_config,
        rng_config=rng_config
    )

    fold_builder = FoldBuilder.FoldBuilder()
    fold = fold_builder.submission_fold()
    list_of_wavs = fold_builder.get_wavs()

    run_name = "pca_100_16_149"
    dump = True
    pred_dfs = []

    for i, config in list(enumerate(configs)):
        print "Building model %d in run %s" % (i, run_name)
        model = build_model(run_name, i, config, list_of_wavs, fold)
        pred_dfs.append(model.pred_df)

        if dump:
            training_utils.dump_model(model)

    blended_df = submission_utils.normalized_even_blend(pred_dfs)
    submission_utils.write_submission_csv(blended_df, 'submission.csv') 

