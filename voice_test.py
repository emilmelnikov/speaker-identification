import voice

import numpy as np
from numpy import testing


def test_melfilterbank():
    got = voice.melfilterbank(1, 5, 1, 0, 1)[0]
    want = [1/3, 2/3, 1, 2/3, 1/3]
    testing.assert_almost_equal(got, np.array(want))


def test_warpsfeatures():
    cases = [
        ([2, 5, 6, 9, 20], 5, [-1.2816, -0.5244, 0, 0.5244, 1.2816]),
        ([2, 9, 5, 20, 6], 3, [-0.9674, 0.9674, -0.9674, 0.9674, 0]),
    ]
    for (signal, warpsize, want) in cases:
        signal = np.array(signal, dtype=float)[:, np.newaxis]
        got = voice.warpfeatures(signal, warpsize)[:, 0]
        testing.assert_almost_equal(got, want, decimal=4)


def test_deltas():
    cases = [
        ([1, 2, 4], 1, [0.5, 1.5, 1.0]),
        ([1, 2, 4, 2], 2, [0.7, 0.5, 0.2, -0.2]),
    ]
    for (values, order, want) in cases:
        got = voice.deltas(values, order)
        testing.assert_almost_equal(got, want, decimal=2)
