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
                    f"{p}-plots/00-all/integrity_accuracy.png"),
                glia_scripts.generate,
                ['-s', '-n 1', '-u 1', p, 'integrity'])
            ran = True
            break

    assert ran

def checkerboard_test(programs_notebook):
    (programs, lab_notebook) = programs_notebook
    ran = False
    for p in programs:
        program = glia.get_experiment_protocol(lab_notebook,p)
        if re.search("checkerboard", program['epl']):
            with Run(lab_notebook) as r:
                r.command(lab_notebook,
                    partial(assert_isfile,
                        f"random_{p}.npz"),
                    glia_scripts.generate,
                    ['-s', '-n 1', '-u 1', p, 'convert', '-c'])
                r.command(lab_notebook,
                    partial(assert_isfile,
                        f"random_{p}-plots/00-all/checkerboard_acuity.png"),
                    glia_scripts.classify_cmd,
                    ['-cs', f"random_{p}.npz"])
                ran = True
            break
    assert ran
#
# def checkerboard_test(programs_notebook):
#     (programs, lab_notebook) = programs_notebook
#     ran = False
#     for p in programs:
#         program = glia.get_experiment_protocol(lab_notebook,p)
#         if re.search("checkerboard", program['epl']):
#             with runner.isolated_filesystem():
#                 lab = 'lab_notebook.yaml'
#                 with open(lab, 'w') as f:
#                     yaml.dump_all(lab_notebook,f)
#                 result = runner.invoke(*invoke, echo=True)
#                 if result.exit_code != 0:
#                     _, error, tb = result.exc_info
#                     print_tb(tb)
#                     raise(error)
#
#                 f_assertions()
#             command(lab_notebook,
#                 partial(assert_isfile,
#                     f"random_{p}.npz"),
#                 glia_scripts.generate,
#                 ['-s', '-n 1', '-u 1', p, 'convert', '-c'])
#             command(lab_notebook,
#                 partial(assert_isfile,
#                     f"{p}-plots/00-all/checkerboard_accuracy.png"),
#                 glia_scripts.classify_cmd,
#                 ['-c', f"random_{p}.npz"])
#             ran = True
#             break
#     assert ran



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
