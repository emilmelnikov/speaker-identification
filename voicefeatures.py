#!/usr/bin/env python


"""Extract voice features from audio files."""


import argparse


import numpy as np
import scipy.fftpack
import scipy.io.wavfile
import scipy.signal
import scipy.stats


def hz2mel(hz):
    """Convert Hertz to Mel."""
    return 1127 * np.log1p(np.asarray(hz) / 700)


def mel2hz(mel):
    """Convert Mel to Hertz."""
    return 700 * np.expm1(np.asarray(mel) / 1127)


def filterbank(nfilters, nsamples, lofreq, hifreq, rate):
    """Create Mel filterbank."""
    lomel = hz2mel(lofreq)
    himel = hz2mel(hifreq)
    melpoints = np.linspace(lomel, himel, nfilters + 2)
    points = mel2hz(melpoints)
    bins = np.asarray(nsamples * points / rate, dtype=int)
    bank = np.zeros([nfilters, nsamples])
    for i in range(nfilters):
        start = bins[i]
        end = bins[i+2]
        bank[i][start:end] = scipy.signal.triang(end - start)
    return bank


def clamp2(i, n, k):
    """For any i in range(0, n), k in range(0, n//2):
    * leave the first k values as range(0, k)
    * change the middle n - 2*k values to k;
    * change the last k values to range(k + 1, 2*k + 1)

    >>> [clamp2(i, 10, 3) for i in range(10)]
    [0, 1, 2, 3, 3, 3, 3, 4, 5, 6]
    """
    return min(i, k) + max(-n+i+k+1, 0)


def getwarps(mfccs, nwarpsamples):
    """Compute warped MFCC coefficients."""

    dist = scipy.stats.norm
    probpoints = (np.arange(nwarpsamples) + 1/2) / nwarpsamples
    lut = dist.pdf(dist.ppf(probpoints))

    nmfccs = mfccs.shape[0]
    warps = np.empty_like(mfccs)

    for i in range(nmfccs):
        windowfrom = np.clip(i - nwarpsamples//2, 0, nmfccs - nwarpsamples)
        window = mfccs[windowfrom:windowfrom+nwarpsamples]
        ranks = np.argsort(window, axis=0)
        irank = clamp2(i, nmfccs, nwarpsamples//2)
        warps[i] = lut[ranks[irank]]

    return warps


def getdeltas(values, size):
    """Compute delta coefficients for values."""
    deltas = np.zeros_like(values)
    for i in range(1, size+1):
        deltas[:i] += i * (values[:i] - values[0])
        deltas[i:-i] += i * (values[2*i:] - values[:-2*i])
        deltas[-i:] += i * (values[-1] - values[-i:])
    norm = 2 * np.sum(np.square(np.arange(1, size+1)))
    return deltas / norm


#pylint: disable=too-many-arguments
def fromfile(name, start=0, end=None, framesize=0.025, stepsize=0.010, warpsize=3,
             deltasize=2, nfilters=40, lofreq=0, hifreq=None, nmfccs=20):
    """Get warped MFCC, delta and delta-delta coefficients from the file."""

    # Get the data.
    rate, signal = scipy.io.wavfile.read(name)
    nsamples = signal.shape[0]

    # Derive default parameters.
    if end is None:
        end = nsamples / rate
    if hifreq is None:
        hifreq = rate / 2

    # Convert seconds to samples.
    startsample = round(start * rate)
    endsample = round(end * rate)
    nframesamples = scipy.fftpack.next_fast_len(round(framesize * rate))
    nstepsamples = round(stepsize * rate)
    nwarpsamples = round(warpsize * rate)

    # Get ROI from the signal.
    signal = signal[startsample:endsample]
    if signal.ndim == 2:
        signal = np.mean(signal, axis=1)

    # Obtain filters.
    window = scipy.signal.hamming(nframesamples)
    bank = filterbank(nfilters, nframesamples//2 + 1, lofreq, hifreq, rate)

    # Subtract frame size from the total duration and round up.
    nframes = (nsamples-nframesamples-1) // nstepsamples + 1

    # Compute MFCCs.
    mfccs = np.zeros([nframes, nmfccs])
    for i in range(nframes):
        frame = signal[i:i+nframesamples]
        _, psd = scipy.signal.periodogram(frame, fs=rate, window=window)
        mfccs[i] = scipy.fftpack.dct(np.log(bank @ psd), n=nmfccs)
import ipdb; ipdb.set_trace()
    # Compute warped MFCCs, delta and delta-delta coefficients.
    warps = getwarps(mfccs, nwarpsamples)
    deltas = getdeltas(warps, deltasize)
    deltas2 = getdeltas(deltas, deltasize)
    return np.hstack([warps, deltas, deltas2])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('file', help='path to the file')
    args = ap.parse_args()
    features = fromfile(args.file)
    print(features.shape)


if __name__ == '__main__':
    main()
