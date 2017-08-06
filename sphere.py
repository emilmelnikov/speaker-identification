#!/usr/bin/env python


"""NIST Sphere file format.

The format does not have any official specification (or it's very hard to find).
However, an informal description can be found at
http://svr-www.eng.cam.ac.uk/reports/ajr/TR192/node11.html
"""


import argparse

import numpy
import scipy.io.wavfile


def parseheader(header_raw):
    keys = {}

    for line_raw in header_raw.split(b'\n'):
        line = line_raw.strip()

        if not line:
            continue
        if line == b'end_head':
            break

        key_bytes, key_type, value_bytes = line.split()
        key = key_bytes.decode()

        if key_type == b'-i':
            keys[key] = int(value_bytes)
        else:
            # key_type == '-sN', where N == len(value)
            keys[key] = value_bytes.decode()

    return keys


def parsedata(fd, header):
    nsamples = header['sample_count']
    nchannels = header['channel_count']
    samplesize = header['sample_n_bytes']
    rawdata = fd.read(nsamples * nchannels * samplesize)

    byteformat = header['sample_byte_format']
    byteformat_table = {'10': '>', '01': '<'}
    dtype = byteformat_table[byteformat] + 'i' + str(samplesize)
    data = numpy.frombuffer(rawdata, dtype=dtype)

    data = data.reshape([nsamples, nchannels])
    return data


def read(fd):
    version_raw = fd.readline()
    version = version_raw.strip().decode()

    headersize_raw = fd.readline()
    headersize = int(headersize_raw.strip())

    remaining = headersize - len(version_raw) - len(headersize_raw)
    header_raw = fd.read(remaining)
    header = parseheader(header_raw)

    data = parsedata(fd, header)
    return version, header, data


def readfile(filename):
    with open(filename, 'rb') as fd:
        return read(fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('from', help='source NIST Sphere file')
    ap.add_argument('to', help='target WAV file')
    args = ap.parse_args()

    version, header, data = readfile(getattr(args, 'from'))

    print('version: {}'.format(version))
    print('header:')
    for key, val in header.items():
        print('    {}: {}'.format(key, val))

    rate = header['sample_rate']
    scipy.io.wavfile.write(args.to, rate, data)


if __name__ == '__main__':
    main()
