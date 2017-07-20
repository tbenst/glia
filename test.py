import matplotlib
import pytest
matplotlib.use('agg')
code = pytest.main(['--doctest-modules'])
exit(code)
