"""Speaker identification."""


import numpy as np

from scipy import fftpack as fft
from scipy import signal
from scipy import stats


def hz2mel(hz):
    """Convert Hertz to Mel."""
    return 1127 * np.log1p(hz / 700)


def mel2hz(mel):
    """Convert Mel to Hertz."""
    return 700 * np.expm1(mel / 1127)


def melfilterbank(nfilters, nsamples, freq, lofreq, hifreq):
    """Create Mel filterbank.

    Parameters
    ----------
    nfilters : int
        Number of Mel filters.
    nsamples : int
        Number of samples per filter.
    freq : int or float
        Frequency (sample rate) of the signal.
    lofreq : int or float
        Left boundary for fitlers.
    higreq : int or float
        Right boundary for fitlers.

    Returns
    -------
    bank : ndarray
        2D array (nfilters, nsamples) of Mel filters.
    """
    lomel = hz2mel(lofreq)
    himel = hz2mel(hifreq)
    melpoints = np.linspace(lomel, himel, nfilters+2)
    points = np.vectorize(mel2hz)(melpoints)
    bins = np.asarray(points * (nsamples/freq), dtype=int)
    bank = np.zeros([nfilters, nsamples])
    for i in range(nfilters):
        start = bins[i]
        end = bins[i+2]
        bank[i][start:end] = signal.triang(end - start)
    return bank


def warpfeatures(values, windowlen):
    """Perform feature warping using the standard normal distribution.

    Arguments
    ---------
    values : array_like
        Feature values.
    windowlen : int
        Warp window length.

    Returns
    -------
    warped : ndarray
        Warped features.
    """

    values = np.asarray(values)
    binvals = (np.arange(windowlen) + 1/2) / windowlen
    lut = stats.norm.ppf(binvals)

    nvalues = values.shape[0]
    nwindows = nvalues - windowlen + 1
    warped = np.zeros_like(values, dtype=float)

    for i in range(nwindows):
        window = values[i:i+windowlen]
        ranks = np.argsort(window, axis=0)
        if i == 0:
            warped[:windowlen//2] = lut[ranks[:windowlen//2]]
        if i == nwindows-1:
            warped[i+windowlen//2+1:] = lut[ranks[windowlen//2+1:]]
        warped[i+windowlen//2] = lut[ranks[windowlen//2]]

    return warped


def deltas(values, order):
    """Compute delta coefficients for values.

    Arguments
    ---------
    values : array_like
        Source values.
    order : int
        Delta order.

    Returns
    -------
    d : ndarray
        Delta values.
    """
    values = np.asarray(values)
    d = np.zeros_like(values, dtype=float)
    for i in range(1, order+1):
        d[:i] += i * (values[i:2*i] - values[0])
        d[i:-i] += i * (values[2*i:] - values[:-2*i])
        d[-i:] += i * (values[-1] - values[-2*i:-i])
    norm = 2 * np.sum(np.square(np.arange(1, order+1)))
    return d / norm


class Signal:
    """Signal with metadata required for speaker identification.

    Attributes
    ----------
    samples : array_like
        1D array with the signal samples.
    freq : int or float
        Frequency (sample rate) of the signal.
    framesecs : int or float
        Duration of a single frame in seconds.
    stepsecs : int or float
        Duration of the inter-frame interval in seconds.
    lofreq : int or float
        Low frequency bound for the MFCC extraction.
    hifreq : int or float
        High frequency bound for the MFCC extraction.
    nfilters : int
        Number of Mel filters.
    nmfccs : int
        Number of MFCCs.
    nsamples : int
        The total number of signal samples.
    framesize : int
        Number of samples in a single frame.
    stepsize : int
        Number of samples between consecutive frames.
    filterbank : ndarray
        Mel filterbank.
    """

    def __init__(self, samples, freq, framesecs, stepsecs, nfilters, nmfccs, lofreq, hifreq):
        """Create a new signal."""
        # Provided parameters.
        self.samples = np.asarray(samples)
        self.freq = freq
        self.framesecs = framesecs
        self.stepsecs = stepsecs
        self.lofreq = lofreq
        self.hifreq = hifreq
        self.nfilters = nfilters
        self.nmfccs = nmfccs
        # Computed parameters.
        self.nsamples = len(samples)
        self.framesize = fft.next_fast_len(int(round(framesecs * freq)))
        self.stepsize = int(round(stepsecs * freq))
        self.nframes = (self.nsamples-self.framesize) // self.stepsize + 1
        self.filterbank = melfilterbank(self.nfilters, self.nsamples,
                                        self.lofreq, self.hifreq, self.freq)
