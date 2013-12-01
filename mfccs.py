# Adapted from https://github.com/jameslyons/python_speech_features
# Original Author: James Lyons 2012
# Modified by Dima Kamalov Nov 2013

import numpy
from scipy.fftpack import dct

import warnings
warnings.filterwarnings('ignore')


def framesig(sig,frame_len,frame_step,winfunc=lambda x:numpy.ones((1,x))):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = round(frame_len)
    frame_step = round(frame_step)
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + numpy.ceil((1.0*slen - frame_len)/frame_step)

    padlen = (numframes-1)*frame_step + frame_len

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig,zeros))

    indices = numpy.tile(numpy.arange(0,frame_len),(numframes,1)) + numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = numpy.array(indices,dtype=numpy.int32)
    frames = padsignal[indices]
    win = numpy.tile(winfunc(frame_len),(numframes,1))
    return frames*win


def deframesig(frames,siglen,frame_len,frame_step,winfunc=lambda x:numpy.ones((1,x))):
    """Does overlap-add procedure to undo the action of framesig.

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round(frame_len)
    frame_step = round(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = numpy.tile(numpy.arange(0,frame_len),(numframes,1)) + numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = numpy.array(indices,dtype=numpy.int32)
    padlen = (numframes-1)*frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = numpy.zeros((1,padlen))
    window_correction = numpy.zeros((1,padlen))
    win = winfunc(frame_len)

    for i in range(0,numframes):
        window_correction[indices[i,:]] = window_correction[indices[i,:]] + win + 1e-15 #add a little bit so it is never zero
        rec_signal[indices[i,:]] = rec_signal[indices[i,:]] + frames[i,:]

    rec_signal = rec_signal/window_correction
    return rec_signal[0:siglen]


def magspec(frames,NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = numpy.fft.rfft(frames,NFFT)
    return numpy.absolute(complex_spec)


def powspec(frames,NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0/NFFT * numpy.square(magspec(frames,NFFT))


def logpowspec(frames,NFFT,norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 1.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames,NFFT);
    ps[ps<=1e-30] = 1e-30
    lps = 10*numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps


def preemphasis(signal,coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return numpy.append(signal[0],signal[1:]-coeff*signal[:-1])


class MFCCs():


    def __init__(self, corner_freq=700.0):
        self.corner_freq = float(corner_freq)


    def compute_mfccs(self, signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
              nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True):
        """Compute MFCC features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param numcep: the number of cepstrum to return, default 13
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
        :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
        :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
        """
        feat,energy = self.fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)
        feat = numpy.log(feat+1e-100)
        feat = dct(feat, type=3, axis=1, norm='ortho')[:,:numcep]
        feat = self.lifter(feat,ceplifter)
        if appendEnergy: feat[:,0] = numpy.log(energy+1e-100) # replace first cepstral coefficient with log of frame energy
        feat = numpy.nan_to_num(feat)
        #occasionally a value will be 0, which means we can't properly take a log
        return feat


    def fbank(self, signal,samplerate=16000,winlen=0.025,winstep=0.01,
              nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97):
        """Compute Mel-filterbank energy features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between seccessive windows in seconds. Default is 0.01s (10 milliseconds)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
            second return value is the energy in each frame (total energy, unwindowed)
        """
        highfreq= highfreq or samplerate/2
        signal = preemphasis(signal,preemph)
        frames = framesig(signal, winlen*samplerate, winstep*samplerate)
        pspec = powspec(frames,nfft)
        energy = numpy.sum(pspec,1) # this stores the total energy in each frame

        fb = self.get_filterbanks(nfilt,nfft,samplerate, lowfreq, highfreq)
        feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
        return feat,energy


    def logfbank(self, signal,samplerate=16000,winlen=0.025,winstep=0.01,
              nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97):
        """Compute log Mel-filterbank energy features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between seccessive windows in seconds. Default is 0.01s (10 milliseconds)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
        """
        feat,energy = self.fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)
        return numpy.log(feat)


    def hz2mel(self, hz):
        """Convert a value in Hertz to Mels

        :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 2595 * numpy.log10(1+hz/self.corner_freq)


    def mel2hz(self, mel):
        """Convert a value in Mels to Hertz

        :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        return self.corner_freq*(10**(mel/2595.0)-1)


    def get_filterbanks(self, nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq= highfreq or samplerate/2

        # compute points evenly spaced in mels
        lowmel = self.hz2mel(lowfreq)
        highmel = self.hz2mel(highfreq)
        melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = numpy.floor((nfft+1)*self.mel2hz(melpoints)/samplerate)

        fbank = numpy.zeros([nfilt,nfft/2+1])
        for j in xrange(0,nfilt):
            for i in xrange(int(bin[j]),int(bin[j+1])):
                fbank[j,i] = (i - bin[j])/(bin[j+1]-bin[j])
            for i in xrange(int(bin[j+1]),int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])

        return fbank


    def lifter(self, cepstra,L=22):
        """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
        magnitude of the high frequency DCT coeffs.

        :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
        :param L: the liftering coefficient to use. Default is 22.
        """
        lift = 1
        if L != 0:
            nframes,ncoeff = numpy.shape(cepstra)
            n = numpy.arange(ncoeff)
            lift += (L/2)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra# original from https://github.com/jameslyons/python_speech_features
