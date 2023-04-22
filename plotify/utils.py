import warnings
import matplotlib as mpl
import subprocess

from distutils.spawn import find_executable
from subprocess import Popen


def usetex(use=True, silent=False, force=False):
    """
    ## Description

    Enables rendering fonts with LaTeX.

    ## Arguments

    * `use`: Disables Latex rendering if False.
    * `silent`: Prints error message if True.
    * `force`: Uses external LaTeX software (eg, texlive, mactex) if True; else, uses matplotlib's implementation.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not use:
            mpl.rc('text', usetex=False)
        elif find_executable('latex'):
            if force or Popen(
                ['kpsewhich', 'type1ec.sty'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            ).communicate()[0]:
                mpl.rc('text', usetex=True)
        elif not silent:
            print('texlive-full is not installed, plotify cannot use tex.')
