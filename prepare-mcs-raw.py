import os
import sys
from subprocess import call

myfile = sys.argv[1]

name, ext = os.path.splitext(myfile)
file = open(myfile, mode='rb')
header_end = "EOH".encode("Windows-1252")
with open(name + ".data", 'wb') as newfile:
    for line in file:
        newfile.write(line.decode("Windows-1252").encode("utf8"))
        if line == b"EOH\r\n":
            break

    newfile.write(file.read())

file.close()

call(["cp", "../spyking-circus/template.params", "{}.params".format(name)])
print("Completed")