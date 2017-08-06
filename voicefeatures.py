#!/usr/bin/env python


"""Extract voice features from audio files."""


import argparse

import numpy
import scipy.fftpack
import scipy.io.wavfile
import scipy.signal
import scipy.stats

import vad


def hz2mel(hz):
    """Convert Hertz to Mel.

    >>> print(round(hz2mel(1000)))
    1000.0
    """
    return 1127 * numpy.log1p(numpy.asarray(hz) / 700)


def mel2hz(mel):
    """Convert Mel to Hertz.

    >>> print(round(mel2hz(1000)))
    1000.0
    """
    return 700 * numpy.expm1(numpy.asarray(mel) / 1127)


def framecount(nsamples, nframesamples, nstepsamples):
    """Compute the number of frames in a sample window.

    >>> framecount(10, 1, 1)
    10
    >>> framecount(10, 2, 1)
    9
    >>> framecount(10, 1, 2)
    5
    >>> framecount(10, 2, 2)
    5
    >>> framecount(10, 3, 2)
    4
    >>> framecount(10, 2, 3)
    3
    >>> framecount(10, 3, 3)
    3
    """
    return (nsamples-nframesamples) // nstepsamples + 1


def filterbank(nfilters, nsamples, lofreq, hifreq, rate):
    """Create Mel filterbank.

    >>> print(*[ round(x, 2) for x in filterbank(1, 5, 0, 1, 1)[0] ])
    0.33 0.67 1.0 0.67 0.33
    """
    lomel = hz2mel(lofreq)
    himel = hz2mel(hifreq)
    melpoints = numpy.linspace(lomel, himel, nfilters + 2)
    points = mel2hz(melpoints)
    bins = numpy.asarray(nsamples * points / rate, dtype=int)
    bank = numpy.zeros([nfilters, nsamples])
    for i in range(nfilters):
        start = bins[i]
        end = bins[i+2]
        bank[i][start:end] = scipy.signal.triang(end - start)
    return bank


def getwarps(values, windowlen):
    """Warp values over the standard normal distribution.

    >>> print(*[ round(x, 4) for x in getwarps(numpy.array( [[2.0, 5, 6, 9, 20]] ).T, 5)[:,0] ])
    -1.2816 -0.5244 0.0 0.5244 1.2816
    >>> print(*[ round(x, 4) for x in getwarps(numpy.array( [[2.0, 9, 5, 20, 6]] ).T, 3)[:,0] ])
    -0.9674 0.9674 -0.9674 0.9674 0.0
    """

    values = numpy.asarray(values)
    binvals = (numpy.arange(windowlen) + 1/2) / windowlen
    lut = scipy.stats.norm.ppf(binvals)

    nvalues = values.shape[0]
    nwindows = framecount(nvalues, windowlen, 1)
    warped = numpy.zeros_like(values)

    for i in range(nwindows):
        window = values[i:i+windowlen]
        ranks = numpy.argsort(window, axis=0)
        if i == 0:
            warped[:windowlen//2] = lut[ranks[:windowlen//2]]
        if i == nwindows-1:
            warped[i+windowlen//2+1:] = lut[ranks[windowlen//2+1:]]
        warped[i+windowlen//2] = lut[ranks[windowlen//2]]

    return warped


def getdeltas(values, size):
    """Compute delta coefficients for values.

    >>> print(*[round(x, 2) for x in getdeltas([1, 2, 4], 1)])
    0.5 1.5 1.0
    >>> print(*[round(x, 2) for x in getdeltas([1, 2, 4, 2], 2)])
    0.7 0.5 0.2 -0.2
    """
    values = numpy.asarray(values)
    deltas = numpy.zeros_like(values)
    for i in range(1, size+1):
        deltas[:i] += i * (values[i:2*i] - values[0])
        deltas[i:-i] += i * (values[2*i:] - values[:-2*i])
        deltas[-i:] += i * (values[-1] - values[-2*i:-i])
    norm = 2 * numpy.sum(numpy.square(numpy.arange(1, size+1)))
    return deltas / norm


def psds(signal, rate, framelen, steplen, window):
    """Compute PSDs from signal frames."""
    nsamples = signal.shape[0]
    nframes = (nsamples - framelen) // steplen + 1
    nfft = framelen//2 + 1
    powers = numpy.empty((nframes, nfft))
    for i in range(0, nframes, steplen):
        frame = signal[i:i+framelen]
        _, power = scipy.signal.periodogram(frame, detrend=False, fs=rate, window=window)
        powers[i] = power
    return powers


def mfccs(powers, rate, nfilters, nmfccs, lofreq, hifreq):
    """Compute MFCCs from PSDs."""
    bank = filterbank(nfilters, nmfccs, lofreq, hifreq, rate)
    return scipy.fftpack.dct(numpy.log(numpy.matmul(powers, bank)), n=nmfccs)


#pylint: disable=too-many-arguments
def fromfile(name, start=0, end=None, framesize=0.025, framestepsize=0.010,
             warpsize=3, deltasize=2, nfilters=40,
             lofreq=0, hifreq=None, nmfccs=20):
    """Get warped MFCC, delta and delta-delta coefficients from the file."""

    # Get the data.
    rate, signal = scipy.io.wavfile.read(name)

    # Derive default parameters.
    if end is None:
        end = signal.shape[0] / rate
    if hifreq is None:
        hifreq = rate / 2

    # Convert seconds to samples.
    startsample = int(round(start * rate))
    endsample = int(round(end * rate))
    nframesamples = scipy.fftpack.next_fast_len(int(round(framesize * rate)))
    nframestepsamples = int(round(framestepsize * rate))
    nwarpfeatures = int(round(warpsize * rate))

    # Get ROI from the signal.
    signal = signal[startsample:endsample]
    if signal.ndim == 2:
        signal = numpy.mean(signal, axis=1)

    nsamples = signal.shape[0]
    nframes = framecount(nsamples, nframesamples, nframestepsamples)
    nbands = nframesamples//2 + 1

    # Compute power spectra for all frames.
    spectra = numpy.zeros([nframes, nbands])
    for i in range(0, nframes, nframestepsamples):
        frame = signal[i:i+nframesamples]
        _, psd = scipy.signal.periodogram(frame, detrend=False, fs=rate, window='hanning')
        spectra[i] = psd

    # Get a boolean mask with True for speech frames.
    vadmask = vad.lstd(spectra)
    nspeechframes = numpy.sum(vadmask)

    # Compute MFCCs.
    mfccs = numpy.zeros([nspeechframes, nmfccs])
    bank = filterbank(nfilters, nbands, lofreq, hifreq, rate)
    for i in range(nspeechframes):
        mfccs[i] = scipy.fftpack.dct(numpy.log(bank @ spectra[i]), n=nmfccs)

    # Compute warped MFCCs, delta and delta-delta coefficients.
    warps = getwarps(mfccs, nwarpfeatures)
    deltas = getdeltas(warps, deltasize)
    deltas2 = getdeltas(deltas, deltasize)
    return numpy.hstack([warps, deltas, deltas2])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('file', help='path to the file')
    ap.add_argument('--start', type=float, default=0, help='start second')
    ap.add_argument('--end', type=float, default=None, help='end second')
    args = ap.parse_args()
    features = fromfile(args.file, start=args.start, end=args.end)


if __name__ == '__main__':
    with suppress(KeyboardInterrupt):
        main()
