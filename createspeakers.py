#!/usr/bin/env python


"""Create speaker WAV files from voxforge archives"""


import argparse
import os
import tarfile

import numpy as np
import scipy.io.wavfile


def dataerror(msg, *args, **kwargs):
    print('\nDATAERROR: ' + msg.format(*args, **kwargs))


def getgender(f):
    for rawline in f:
        line = rawline.strip().lower()
        if line.find(b'gender') == -1:
            continue
        if line.find(b'female') != -1:
            return 'female'
        if line.find(b'male') != -1:
            return 'male'
    return ''


def extractwav(archivename):
    joinedrate = None
    gender = None
    sigs = []

    with tarfile.open(archivename) as tar:
        for member in tar.getmembers():
            filename = member.name
            fullname = os.path.join(archivename, filename)

            if filename.endswith('README'):
                if gender is not None:
                    dataerror('{}: multiple READMEs: last with path {}',
                              archivename, filename)
                    return 0, np.empty(), ''
                with tar.extractfile(member) as f:
                    gender = getgender(f)

            if filename.endswith('.wav'):
                with tar.extractfile(member) as f:
                    rate, sig = scipy.io.wavfile.read(f)

                    if joinedrate is None:
                        joinedrate = rate

                    if rate != joinedrate:
                        dataerror('{}: unequal signal rates: got {}, expected {}',
                                  fullname, rate, joinedrate)
                        return 0, np.empty(), ''

                    if sigs and sig.ndim != sigs[-1].ndim:
                        dataerror('{}: unequal channel count: got {}, expected {}',
                                  fullname, sig.ndim, sigs[-1].ndim)
                        return 0, np.empty(), ''

                    sigs.append(sig)

    if gender not in ['male', 'female']:
        dataerror('{}: unknown or undefined gender: {}', archivename, repr(gender))
        return 0, np.empty(0), ''

    if not sigs:
        dataerror('{}: no WAV files', archivename)
        return 0, np.empty(0), ''

    return joinedrate, np.concatenate(sigs), gender


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archivelist', help='file with archive paths, one per line')
    ap.add_argument('-p', '--prefix', help='output directory prefix', default='.')
    args = ap.parse_args()

    for gender in ['male', 'female']:
        dirname = os.path.join(args.prefix, gender)
        if not os.path.exists(dirname):
            os.makedirs(dirname, mode=0o755)

    archivepaths = []
    with open(args.archivelist) as f:
        for rawline in f:
            line = rawline.strip()
            if not line:
                continue
            archivepaths.append(line)

    width = len(str(len(archivepaths)))
    progressfmt = '\rProcessing archive [{{:{0}}}/{{:{0}}}] [{{:6.2f}}%]'.format(width)

    njoined = 0

    for i, archivename in enumerate(archivepaths, 1):
        print(progressfmt.format(i, len(archivepaths), i/len(archivepaths)*100), end='')

        try:
            rate, sig, gender = extractwav(archivename)
            if not rate:
                continue
            origfilename = os.path.split(archivename)[-1]
            newfilename = origfilename.rsplit('.', 1)[0] + '.wav'
            fullname = os.path.join(args.prefix, gender, newfilename)
            scipy.io.wavfile.write(fullname, rate, sig)
            njoined += 1

        except Exception as e:
            print('\nEXCEPTION: {}'.format(e))

    print(('\nStats: total = {}, discarded = {}, retained = {} ({:6.2f}%)'.
           format(len(archivepaths),
                  len(archivepaths) - njoined,
                  njoined,
                  njoined/len(archivepaths)*100)))

if __name__ == '__main__':
    main()
