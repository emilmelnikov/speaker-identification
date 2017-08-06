#!/bin/sh

# Download voices from voxforge.org.
# Consider running this script inside tmux.

# Approximate filename pattern: {speakerid}-{yyyymmdd}-{suffix}.tgz

URL='http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'

DIRPREFIX='data/voxforge'

mkdir -p "$DIRPREFIX"

wget "$URL" \
	--recursive \
	--level 1 \
	--accept '*.tgz' \
	--no-directories \
	--directory-prefix "$DIRPREFIX" \
	--no-clobber
