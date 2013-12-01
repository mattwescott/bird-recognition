import numpy as np
import pandas as pd

import sklearn.decomposition

import mfccs
import submission_utils
import wav_utils


class GenerateSpectrograms:

    def __init__(self, sr=44100, nstep=int(44100*0.005), nwin=int(44100*0.01)):
        self.set_params(sr=sr, nstep=nstep, nwin=nwin)

    def set_params(self, **params):
        self.params = params

    def get_params(self):
        return self.params

    def transform(self, list_of_wavs):
        return [wav_utils.compute_spectrogram(wav_array) for wav_array in list_of_wavs]


class GenerateMFCCs:

    def __init__(self, **params):

        self.set_params(**params)
        self.mfccs = mfccs.MFCCs(self.get_params()['cornerfreq'])

    def set_params(self, **params):

        framelen = 256./44100
        default_config = dict(
            framelen = framelen,
            framestep = framelen/3,
            numcep = 14,
            nfilt = 21,
            nfft = 256,
            lowfreq = 500,
            highfreq = None,
            preemph = 0.97,
            ceplifter = 22,
            appendEnergy = True,
            cornerfreq = 1500.0
       )

        config = default_config
        config.update(params)

        self.params = config

    def get_params(self):
        return self.params

    def wav_array_to_mfccs(self, wav_array):

        config = dict(
            signal = wav_array,
            samplerate = 44100,
            winlen = self.get_params()['framelen'],
            winstep = self.get_params()['framestep'],
            numcep = self.get_params()['numcep'],
            nfilt = self.get_params()['nfilt'],
            nfft = self.get_params()['nfft'],
            lowfreq = self.get_params()['lowfreq'],
            highfreq = self.get_params()['highfreq'],
            preemph = self.get_params()['preemph'],
            ceplifter = self.get_params()['ceplifter'],
            appendEnergy = self.get_params()['appendEnergy']
        )

        return self.mfccs.compute_mfccs(**config)

    def transform(self, list_of_wavs):
        return [self.wav_array_to_mfccs(wav_array) for wav_array in list_of_wavs]


class WindowMFCCs:

    def __init__(self, window_length=75, overlap=0.8):
        self.set_params(window_length=window_length, overlap=overlap)

    def set_params(self, **params):
        self.params = params

    def get_params(self):
        return self.params

    def split_into_windows(self, mfcc_array):
        #input: mfcc array of shape (n_s samples, k coefficients)
        #output: window array of shape (n_w windows, n_f frames, k coefficients)

        window_length = self.get_params()['window_length']
        overlap = self.get_params()['overlap']

        windows = []
        start_frame = 0
        while start_frame+window_length < len(mfcc_array):
            windows.append(mfcc_array[start_frame:start_frame+window_length])
            start_frame += int(window_length*(1-overlap))

        return np.array(windows)

    def transform(self, list_of_frames):
        return [self.split_into_windows(mfcc_array) for mfcc_array in list_of_frames]


class SummarizeWindowedMFCCs:

    def __init__(self, summary_function=np.mean):
        self.set_params(summary_function=summary_function)

    def set_params(self, **params):
        self.params = params

    def get_params(self):
        return self.params

    def summarize_window(self, window):
        return np.apply_along_axis(self.get_params()['summary_function'], 1, window)

    def transform(self, list_of_windows):
        return [self.summarize_window(window) for window in list_of_windows]


class PCASummarizeWindowedMFCCs:

    def __init__(self, pca_components=3, summary_function=np.mean):
        self.set_params(pca_components=pca_components, summary_function=summary_function)
        self.pca_builder = lambda: sklearn.decomposition.PCA(n_components=pca_components)

    def set_params(self, **params):
        self.params = params

    def get_params(self):
        return self.params

    def summarize_window(self, window):
        return np.apply_along_axis(self.get_params()['summary_function'], 1, window)

    def transform(self, list_of_windows):

        n_mfccs = list_of_windows[0].shape[2]
        # We will be running one PCA decomposition for each of the ~14 mfccs
        self.pcas = [self.pca_builder() for _ in range(n_mfccs)]

        # Remember the splits so we can separate the windows back into wavs
        splits = np.cumsum([windows.shape[0] for windows in list_of_windows])[:-1]
        # windows will have shape: Nw (windows total in all clips) x Nf (mfcc
        # frames in each window) x Nm (mfcc coefficients)
        windows = np.vstack(list_of_windows)

        summarized_mfcc_arrays = []

        # Now we run a PCA decomposition for each mfcc coefficient, finding a
        # basis for the arrays of windowed mfcc frames
        # Each window will be summarized by Nm (mfcc coefficients) x
        # pca_components
        for i, mfcc_array in enumerate(np.transpose(windows, axes=[2, 0, 1])):
            summarized_mfcc_arrays.append(self.pcas[i].fit_transform(mfcc_array))

        n_pca = self.get_params()['pca_components']

        # Flatten the Nm x pca_component array for each window, and split back # into wavs
        return np.split(np.dstack(summarized_mfcc_arrays).reshape((-1, n_mfccs*n_pca)), splits)



class CollapseWindowPredictions:

    def __init__(self, collapse_function=np.mean):
        self.set_params(collapse_function=collapse_function)

    def set_params(self, **params):
        self.params = params

    def get_params(self):
        return self.params

    def transform(self, list_of_window_predictions):
        return np.array([np.apply_along_axis(self.get_params()['collapse_function'], 0, prediction_array) for prediction_array in list_of_window_predictions])


def make_prediction_df(list_of_preds, fold):
    """Make a prediction df in the same format as data.label_df from a fold whose list_of_features are the predictions"""

    return pd.DataFrame(np.array(list_of_preds), index=fold.clipnames, columns=fold.call_ids).iloc[fold.testing_mask]


def score(fold):

    test_clipnames = fold.clipnames[fold.testing_mask]
    pred_df = fold.list_of_features.ix[test_clipnames]
    true_df = fold.label_df.ix[test_clipnames]
    return submission_utils.evaluate(pred_df, true_df)
