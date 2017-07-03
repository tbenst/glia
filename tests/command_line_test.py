import click
import requests
import asyncio
import glia_scripts
import os
from traceback import print_tb
from bs4 import BeautifulSoup
from click.testing import CliRunner

import functools
import pytest

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


eyecandy_url = 'http://localhost:3000'
def get_programs():
    s = requests.Session()
    index = s.get(eyecandy_url)
    soup = BeautifulSoup(index.content)
    raw_programs = soup.select("select[name=program] option")
    programs = list(filter(lambda x: x!="custom",
                    [p["value"] for p in raw_programs]))
    s.post(eyecandy_url + '/window',
                      headers={
                           'windowHeight': "1140",
                           'windowWidth': "912",
                           })
    lab_notebook = ""
    for p in programs:

        r = s.post(eyecandy_url + '/start-program',
                         data={
                              'filename': p,
                              'program': p,
                              'seed': "12345",
                              'submitButton': 'start',
                              })

        if r.status_code != 200:
            raise(ValueError(f"Internal Server Error for {p}"))
        lab_notebook+=r.text
    return (programs, lab_notebook)

def eyecandy_lab_notebook_test():
    (programs, lab_notebook) = get_programs()
    for p in programs:
        runner = cli()
        with runner.isolated_filesystem():
            lab = 'lab_notebook.yaml'
            with open(lab, 'w') as f:
                f.write(lab_notebook)
                f.write('\n')
            result = runner.invoke(glia_scripts.generate,
                ['-s', '-n 1', '-u 1', p, 'integrity'])
            if result.exit_code != 0:
                _, error, tb = result.exc_info
                print_tb(tb)
                raise(error)
            assert os.path.isfile(f"{p}-plots/00-all/integrity-accuracy.png")
