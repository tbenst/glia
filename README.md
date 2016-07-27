# Glia
A package to support neuroscientists in analyzing MEAs.

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