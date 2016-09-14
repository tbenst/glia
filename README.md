# Glia
A package to support neuroscientists in analyzing MEAs.


## How to analyze data:
1. Convert *.mcd files into *.voltages

  1.	Pull files from MEA computer onto local machine 

  2.	Open docker and go to folder with data
e.g. `docker run -v /c/Users/sandt/Desktop/160913:/data tbenst/mcd`
-> this will automatically start the conversion process in the folder
Wait until all files are converted, i.e. the terminal says: process finished!

3.	Find out header length

    Open new terminal:
    `chdir /Documents/Github/`
	run: 
    `docker run --rm -v /c/Users/Administrator/OneDrive/jupyter-notebooks:/notebooks --link eyecandy_web_1:eyecandy -p 8888:8888 tbenst/jupyter-neuro`
	go to Chrome and type: `localhost:8888`
	go to “get header offset”
	type in folder with *.voltages file
	run script (last line will spit out header length)

2. spike sorting
  4.	Load data for spike sorting
Open Plexon:
File -> import data -> import binary file
Open file location
	60 channels
Sampling frequency (usually 25000, information can also be found in header)
	Header length (see point 3)
	Press ok

  5.	Filter data and detect spikes
Open “waveforms”
Filter continuous data
	Butterworth 4th order, 330 Hz 
		For all channels
Detect spikes:
	Open “waveforms” -> detect spikes
		Threshold -3.8 -> for all channels

  6.	Spike sorting
Open “sort”
Perform automatic sorting 
    `E-M: 15`
 
 when finished: visually check the sorted data and `invalidate` noise spikes
  save waveformes as: *.txt file; all units in one file. delimiter `,`

  select: channel (raw), unit, timestamp


## Dev notes
r = requests.post(base_url + '/analysis/start-program', data={'programYAML': program_yaml})

sid = r.text.

program_yaml = """- with_nested:
    - [50, 150]
    - [750, 1250]
    - [0, PI/4, PI/2, 3*PI/4, PI, 5*PI/4, 3*PI/2, 7*PI/4, 5*PI/4, 3*PI/4, 3*PI/2, PI/2, 7*PI/4, PI/4, 0, PI]
  list:
    - wait:
        time: 1
    - solid:
        time: 1
        backgroundColor: white
    - wait:
        time: 2
    - bar:
       width: items[0]
       speed: items[1]
       angle: items[2]
       barColor: white
       backgroundColor: black
"""

base_url = "http://192.168.29.122:3000"

r = requests.post(base_url + '/analysis/start-program', data={'programYAML': program_yaml})

sid = r.text

r = requests.post(base_url + '/analysis/start-program', data={'programYAML': program_yaml})

sid2 = r.text

n = requests.get(base_url + '/analysis/program/{}'.format(sid))

n.text
