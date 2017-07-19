import cProfile
import os, sys
from optparse import OptionParser

import cProfile, pstats, io
from io import StringIO
import traceback


def main():
    glia_path = os.path.dirname(os.path.realpath(__file__))+'/command_line.py'

    usage = "glia-profile [-o output_file_path] [-s sort] [arg] ..."
    parser = OptionParser(usage=usage)
    parser.allow_interspersed_args = False
    parser.add_option('-o', '--outfile', dest="outfile",
        help="Save stats to <outfile>", default=None)
    parser.add_option('-s', '--sort', dest="sort",
        help="Sort order when printing to stdout, based on pstats.Stats class",
        default=-1)

    (options, args) = parser.parse_args()
    sys.argv[:] = ['glia'] + args

    if len(args) >= 0:
        progname = 'command_line.py'
        # sys.path.insert(0, os.path.dirname(progname))
        with open(glia_path, 'rb') as fp:
            code = compile(fp.read(), progname, 'exec')
        globs = {
            '__file__': progname,
            '__name__': '__main__',
            '__package__': None,
        }

        pr = cProfile.Profile()
        pr.enable()
        try:
            exec(code, globs, None,)
        except Exception as exception:
            traceback.print_tb(exception.__traceback__)
            print(exception)
            pass
        pr.disable()

        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(20)
        print(s.getvalue())

    else:
        parser.print_usage()

if __name__ == '__main__':
    main()
