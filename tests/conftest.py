import sys
from os.path import dirname as d
from os.path import abspath
import pytest
import os

root_dir = d(d(abspath(__file__)))
sys.path.append(root_dir)


if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value
