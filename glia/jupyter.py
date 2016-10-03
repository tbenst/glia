import reprlib

r = reprlib.Repr()
r.maxlist = 10       # max elements displayed for lists
r.maxstring = 100    # max characters displayed for strings
r.maxlevel = 5

def safe_print(*args):
    for arg in args:
        print(r.repr(arg))