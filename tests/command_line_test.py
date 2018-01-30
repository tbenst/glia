import glia_scripts
import glia
import os
import re
import sys
import yaml
from traceback import print_tb
from functools import partial
from lib import *

def integrity_test(programs_notebook):
    (programs, lab_notebook) = programs_notebook
    ran = False
    for p in programs:
        program = glia.get_experiment_protocol(lab_notebook,p)
        if re.search("measureIntegrity", program['epl']):
            command(lab_notebook,
                partial(assert_isfile,
                    f"random_{p}-plots/00-all/integrity_accuracy.png"),
                glia_scripts.generate,
                ['-s', '-n 1', '-u 1', p, 'integrity'])
            ran = True
            break

    assert ran

def checkerboard_test(programs_notebook):
    (programs, lab_notebook) = programs_notebook
    ran = False
    for i,p in enumerate(programs):
        if re.search("checkerboard", p):
            with Run(lab_notebook) as r:
                r.command(lab_notebook,
                    partial(assert_isfile,
                        f"random_{p}.npz"),
                    glia_scripts.generate,
                    ['-n 1', '-u 1', p, 'convert'])
                r.command(lab_notebook,
                    partial(assert_isfile,
                        f"random_{p}-plots/00-all/{p}_acuity.png"),
                    glia_scripts.classify_cmd,
                    ["-s","-d 2", f"random_{p}.npz"])
                ran = True
    assert ran

def grating_test(programs_notebook):
    (programs, lab_notebook) = programs_notebook
    ran = False
    for i,p in enumerate(programs):
        if re.search("grating", p):
            with Run(lab_notebook) as r:
                r.command(lab_notebook,
                    partial(assert_isfile,
                        f"random_{p}.npz"),
                    glia_scripts.generate,
                    ['-n 1', '-u 1', p, 'convert'])
                r.command(lab_notebook,
                    partial(assert_isfile,
                        f"random_{p}-plots/00-all/{p}_acuity.png"),
                    glia_scripts.classify_cmd,
                    ["-s", "-d 2", f"random_{p}.npz"])
                ran = True
    assert ran


def acuity_test(programs_notebook):
    (programs, lab_notebook) = programs_notebook
    ran = False
    for i,p in enumerate(programs):
        if re.search("acuity", p):
            with Run(lab_notebook) as r:
                r.command(lab_notebook,
                    partial(assert_has_at_least_x_files,
                        f"random_{p}-plots/random_{p}_(0, 0)_0",
                        2),
                    glia_scripts.generate,
                    ['-n 1', '-u 1', p, 'acuity'])
                ran = True
    assert ran

def bar_test(programs_notebook):
    (programs, lab_notebook) = programs_notebook
    ran = False
    for i,p in enumerate(programs):
        if re.search("acuity", p):
            with Run(lab_notebook) as r:
                r.command(lab_notebook,
                    partial(assert_has_at_least_x_files,
                        f"random_{p}-plots/random_{p}_(0, 0)_0",
                        1),
                    glia_scripts.generate,
                    ['-n 1', '-u 1', p, 'bar'])
                ran = True
    assert ran

def wedge_test(programs_notebook):
    (programs, lab_notebook) = programs_notebook
    ran = False
    for i,p in enumerate(programs):
        if re.search("wedge", p):
            with Run(lab_notebook) as r:
                r.command(lab_notebook,
                    partial(assert_has_at_least_x_files,
                        f"random_{p}-plots/random_{p}_(0, 0)_0",
                        1),
                    glia_scripts.generate,
                    ['-n 1', '-u 1', p, 'solid'])
                ran = True
    assert ran

def kinetics_test(programs_notebook):
    (programs, lab_notebook) = programs_notebook
    ran = False
    for i,p in enumerate(programs):
        if re.search("kinetic", p):
            with Run(lab_notebook) as r:
                r.command(lab_notebook,
                    partial(assert_has_at_least_x_files,
                        f"random_{p}-plots/random_{p}_(0, 0)_0",
                        1),
                    glia_scripts.generate,
                    ['-n 1', '-u 1', p, 'solid'])
                ran = True
    assert ran


def command(lab_notebook, f_assertions, *invoke):
    "test a command"
    runner = cli()
    with runner.isolated_filesystem():
        lab = 'lab_notebook.yaml'
        with open(lab, 'w') as f:
            yaml.dump_all(lab_notebook,f)
        result = runner.invoke(*invoke, echo=True)
        if result.exit_code != 0:
            _, error, tb = result.exc_info
            print_tb(tb)
            raise(error)

        f_assertions()
#
# def eyecandy_lab_notebook_test(programs_notebook):
#     (programs, lab_notebook) = programs_notebook
#
#     for p in programs:
#         runner = cli()
#         with runner.isolated_filesystem():
#             lab = 'lab_notebook.yaml'
#             with open(lab, 'w') as f:
#                 f.write(lab_notebook)
#                 f.write('\n')
#             result = runner.invoke(glia_scripts.generate,
#                 ['-s', '-n 1', '-u 1', p, 'integrity'])
#             if result.exit_code != 0:
#                 _, error, tb = result.exc_info
#                 print_tb(tb)
#                 raise(error)
#             assert os.path.isfile(f"{p}-plots/00-all/integrity_accuracy.png")
