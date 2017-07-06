# Glia
[![Build Status](https://travis-ci.org/tbenst/glia.svg?branch=master)](https://travis-ci.org/tbenst/glia)

A package to support neuroscientists in analyzing MEAs.


## How to analyze data:
1. Convert *.mcd files into *.voltages

  1. Pull files from MEA computer onto local machine

  2. Open docker and go to folder with data
e.g. `docker run -v /c/Users/sandt/Desktop/160913:/data tbenst/mcd`
-> this will automatically start the conversion process in the folder
Wait until all files are converted, i.e. the terminal says: process finished!

3. Find out header length

    Open new terminal:
    `chdir /Documents/Github/`
    run:
    `docker run --rm -v /c/Users/Administrator/OneDrive/jupyter-notebooks:/notebooks --link eyecandy_web_1:eyecandy -p 8888:8888 tbenst/jupyter-neuro`
    go to Chrome and type: `localhost:8888`
    go to “get header offset”
    type in folder with *.voltages file
    run script (last line will spit out header length)

2. spike sorting
  4. Load data for spike sorting
Open Plexon:
File -> import data -> import binary file
Open file location
    60 channels
Sampling frequency (usually 25000, information can also be found in header)
    Header length (see point 3)
    Press ok

  5. Filter data and detect spikes
Open “waveforms”
Filter continuous data
    Butterworth 4th order, 330 Hz
        For all channels
Detect spikes:
    Open “waveforms” -> detect spikes
        Threshold -3.8 -> for all channels

  6. Spike sorting
Open “sort”
Perform automatic sorting
    `E-M: 15`

 when finished: visually check the sorted data and `invalidate` noise spikes
  save waveformes as: *.txt file; all units in one file. delimiter `,`

  select: channel (raw), unit, timestamp


## Dev notes
sudo docker run -it -v $(pwd):/data tbenst/glia:acuity analyze -v -p 4 -e http://localhost:3000 /data/R1_E1_AMES_50min_acuity.txt integrity solid --wedge bar --by acuity acuity
