import functools
import click
import sys
import os
import yaml
from traceback import print_tb
from click.testing import CliRunner


def assert_within(a,b,within=1):
    assert abs(a-b) <= within

def assert_isfile(filename):
    try:
        assert os.path.isfile(filename)
    except:
        print(os.listdir())
        directory, name = os.path.split(filename)
        if os.path.isdir(directory): print(os.listdir(directory))
        raise

def cli():
    """Yield a click.testing.CliRunner to invoke the CLI."""
    class_ = click.testing.CliRunner

    def invoke_wrapper(f):
        """Augment CliRunner.invoke to emit its output to stdout.

        This enables pytest to show the output in its logs on test
        failures.

        """
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            echo = kwargs.pop('echo', False)
            result = f(*args, **kwargs)

            if echo is True:
                sys.stdout.write(result.output)

            return result

        return wrapper

    class_.invoke = invoke_wrapper(class_.invoke)
    cli_runner = class_()

    return cli_runner

class Run:
    def __init__(self, lab_notebook):
        self.lab_notebook = lab_notebook
        self.runner = cli()

    def __enter__(self):
        #ttysetattr etc goes here before opening and returning the file object
        self.fs = self.runner.isolated_filesystem()
        self.fs.__enter__()
        lab = 'lab_notebook.yaml'
        with open(lab, 'w') as f:
            yaml.dump_all(self.lab_notebook,f)

        return self
    def __exit__(self, type, value, traceback):
        #Exception handling here
        self.fs.__exit__(type, value, traceback)

    def command(self, lab_notebook, f_assertions, *invoke):
        "test a command"
        result = self.runner.invoke(*invoke, echo=True)
        if result.exit_code != 0:
            _, error, tb = result.exc_info
            print_tb(tb)
            raise(error)

        f_assertions()
        return result
