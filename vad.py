"""Voice activity detection."""

import numpy
import scipy.io.wavfile


def boundaries(values):
    """Find boundaries of the non-zero values.

    >>> boundaries(numpy.array([]))
    []
    >>> boundaries(numpy.array([0]))
    []
    >>> boundaries(numpy.array([1]))
    [(0, 1)]
    >>> boundaries(numpy.array([1, 1, 1, 0, 0, 1, 1]))
    [(0, 3), (5, 7)]
    >>> boundaries(numpy.array([0, 1, 0, 1, 0]))
    [(1, 2), (3, 4)]
    """
    nz = numpy.flatnonzero(values)
    if not nz.size:
        return []
    breaks = numpy.flatnonzero(numpy.ediff1d(nz) > 1) + 1
    runs = numpy.split(nz, breaks)
    return [(r[0], r[-1]+1) for r in runs]


def lstd(x, n=6, k=3, alpha=0.95, offset=5, hangover_frames=8,
               hangover_threshold=25, gamma_0=6.0, gamma_1=2.5, e_0=30, e_1=50):
    """Detect speech frames using LSTD algorithm.

    Ramı́rez, J., Segura, J. C., Benı́tez, C., de la Torre, Á., & Rubio, A. (2004).
    Efficient voice activity detection algorithms using long-term speech information.
    Speech Communication, 42(3–4), 271–287.
    https://doi.org/10.1016/j.specom.2003.10.002

    Parameters
    ----------
    x : ndarray
        2d array of power spectra for frames
    n : int
        LSTD window size
    k : int
        noise update window size
    alpha : float
        noise update decay factor
    hangover_frames : int
        number of frames for which non-speech transition is delayed
    hangover_threshold : int or float
        disable hangover if LSTD goes beyond this threshold
    gamma_0 : int or float
        threshold for the clean conditions
    gamma_1 : int or float
        threshold for the noisy conditions
    e_0 : int or float
        energy level for clean conditions
    e_1 : int or float
        energy level for noisy conditions

    Return
    ------
    vadmask : ndarray of bool
        array with the speech frames marked as True
    """
    if x.ndim != 2:
        raise ValueError('frames.ndim != 2')
    nframes, nbands = x.shape

    # Result.
    mask = numpy.zeros(nframes, dtype=bool)

    noise = numpy.mean(x[:n], axis=0)
    e = numpy.sum(noise) * n

    # Compute threshold.
    if e <= e_0:
        gamma = gamma_0
    elif e < e_1:
        gamma_d = gamma_0 - gamma_1
        gamma = e*gamma_d/(e_0-e_1) + gamma_0 - gamma_d/(1-e_1/e_0)
    else:
        gamma = gamma_1
    gamma += offset

    nonspeech_frames = 0
    for i in range(n, nframes-n):
        # Compute initial decision.
        lste = numpy.max(x[i-n:i+n+1], axis=0)
        lstdval = 10 * numpy.log10(numpy.sum(numpy.square(lste/noise)) / nbands)
        mask[i] = lstdval > gamma
        # Hangover (disabled on low noise frames).
        if mask[i] or lstdval > hangover_threshold:
            nonspeech_frames = 0
        elif nonspeech_frames < hangover_frames:
            nonspeech_frames += 1
            mask[i] = True
        # Noise estimation update.
        if mask[i]:
            noise_k = numpy.sum(x[i-k:i+k+1], axis=0) / (2*k+1)
            noise = alpha*noise + (1-alpha)*noise_k

    return mask
