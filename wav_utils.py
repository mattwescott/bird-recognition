import numpy as np

# from IPython.core.display import HTML
# # from http://nbviewer.ipython.org/5507501/the%20sound%20of%20hydrogen.ipynb
# def wavPlayer(filepath):
#     """ will display html 5 player for compatible browser
# 
#     Parameters :
#     ------------
#     filepath : relative filepath with respect to the notebook directory ( where the .ipynb are not cwd)
#                of the file to play
# 
#     The browser need to know how to play wav through html5.
# 
#     there is no autoplay to prevent file playing when the browser opens
#     """
# 
#     src = """
#     <head>
#     <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
#     <title>Simple Test</title>
#     </head>
# 
#     <body>
#     <audio controls="controls" style="width:600px" >
#       <source src="files/%s" type="audio/wav" />
#       Your browser does not support the audio element.
#     </audio>
#     </body>
#     """%(filepath)
#     return HTML(src)
# 

# Adapted from https://mail.python.org/pipermail/chicago/2010-December/007314.html
def compute_spectrogram(wav_array):
    """
    Compute and display a spectrogram.
    """

    sr,x = 44100, wav_array

    ## Parameters: 10ms step, 30ms window
    nstep = int(sr * 0.005)
    nwin  = int(sr * 0.01)
    nfft = nwin

    window = np.hamming(nwin)

    ## will take windows x[n1:n2].  generate
    ## and loop over n2 such that all frames
    ## fit within the waveform
    nn = range(nwin, len(x), nstep)

    X = np.zeros( (len(nn), nfft/2) )

    for i,n in enumerate(nn):
        xseg = x[n-nwin:n]
        z = np.fft.fft(window * xseg, nfft)
        X[i,:] = np.log(np.abs(z[:nfft/2]))

    return np.fft.fftfreq(nwin, 1.0/sr)[:nfft/2], X


# import matplotlib.pyplot as plt
# def plot_spectrogram(freqs, X, syllable_splits=None, low_cutoff_khz=0.5, high_cutoff_khz=10):
# 
#     low_cutoff_index = int(float(len(freqs))/freqs[-1]*1000*low_cutoff_khz)
#     high_cutoff_index = int(float(len(freqs))/freqs[-1]*1000*high_cutoff_khz)
# 
#     if syllable_splits is not None:
#         for lower, upper in syllable_splits:
#             plt.vlines(lower, 0, high_cutoff_khz, linestyles=['solid'], colors=['black'])
# 
#             plt.vlines(upper, 0, high_cutoff_khz, linestyles=['dashed'], colors=['black'])
# 
#     plt.imshow(X.T[low_cutoff_index:high_cutoff_index,:], interpolation='nearest',
#         origin='lower',
#         aspect='auto',
#         extent=[0, X.shape[0], freqs[0], high_cutoff_khz]
#         )
# 
#     plt.show()
