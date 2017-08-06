#!/usr/bin/env python


"""Create Universal Background Model (UBM)."""


import argparse
import os

import sphere


def getvoices(rootdir):
    """Gather voice data from subdirectories of the root directory.

    Each voice has multiple data files in NIST Sphere format.
    Voice IDs are the directory names at the lowest hierarchy level.
    Return a dict from voice IDs to lists of voice data.
    """
    voices = {}
    for dirpath, dirnames, filenames in os.walk(rootdir):
        # Current directory is an intermediate one: skip it.
        if dirnames:
            continue
        voiceid = os.path.split(dirpath)[1]
        # Fail on voices with identical IDs.
        if voiceid in voices:
            raise Exception('duplicate voice ID {}'.format(voiceid))
        voices[voiceid] = []
        for filename in filenames:
            # Skip auxiliary files.
            if not filename.endswith('.WAV'):
                continue
            filepath = os.path.join(dirpath, filename)
            _, _, voicedata = sphere.readfile(filepath)
            voices[voiceid].append(voicedata)
        # Remove the current voice if it does not have any data.
        if not voices[voiceid]:
            del voices[voiceid]
    return voices


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dir', help='directory with speakers, one per directory')
    args = ap.parse_args()
    # makeubm(args.dir)


if __name__ == '__main__':
    main()
