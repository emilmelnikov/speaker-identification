# Data preparation pipeline

## Download data archives from voxforge

```sh
scripts/voxforge-dl.sh
```

It downloads files to `data/voxforge` directory, skipping already downloaded files.
Note that "already downloaded" files might be only partially downloaded.
As of June 2017, total dataset size is 10 GB.

## Create file list for processing

```sh
find data/voxforge -type f > data/archivelist.txt
```

## Create speaker-specific files

```sh
time python createspeakers.py --prefix data data/archivelist.txt
```

This creates `data/male` and `data/female` directories with combined WAV files for each speaker.
Erroneous archives, ones without gender labels and WAV files, will be skipped.
As of June 2017, the resulting files occupy 12 GB: male voices occupy 9.9 GB, female voices occupy 1.8 GB.

After creating combined WAV files, downloaded archives can be deleted.
