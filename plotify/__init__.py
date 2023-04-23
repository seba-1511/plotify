# -*- coding=utf-8 -*-

from ._version import __version__
from .plot import *
from .utils import usetex
from .colors import Vibrant, Maureen, lighten_color
from .custom_plots import Plot, PublicationPlot, LowResPlot, ListContainer, ModernPlot

from . import utils
from . import fonts
from . import colors
from . import markers
from . import custom_plots
from . import smoothing
from . import wandb_plots
from . import tensorboard_plots
