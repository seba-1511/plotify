#!/usr/bin/env python3

import os
import numpy as np
import copy
import statistics as stats
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D #required for 3D plots
from matplotlib import image as mpimg
from matplotlib.patches import Circle, Rectangle, Arrow, FancyBboxPatch, BoxStyle
from matplotlib import patheffects
from matplotlib import font_manager

from tempfile import gettempdir
from collections.abc import Iterable
from itertools import cycle
from time import time

from .utils import usetex
from .colors import MAUREENSTONE_COLORS, Vibrant, LIGHT_GRAY
from .markers import Marker, MARKERS

# high-definition images in IPython notebooks
try:
    if hasattr(__builtins__,'__IPYTHON__'):
        from IPython.display import set_matplotlib_formats
        set_matplotlib_formats('retina')
except:
    pass

try:
    from io import BytesIO as BytesIO
except:
    pass

MEM_IMG = BytesIO()


FONT_SIZE = 20
# MARKERS = ['o', 'X', 'v', '*', 'd', 's', '*', 'p']
PALETTE_NAME = MAUREENSTONE_COLORS
COLORMAP_3D = 'YlGnBu'
TEMP_FILENAME = os.path.join(gettempdir(), 'plotify')
TEMP_FILE_EXT = '.png'

mpl.style.use('seaborn-poster')

# Latex font
#mpl.rc('text', usetex=True)  # Renders math with latex program (slow)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'cm'  # Font for tex


def set_box_color(canvas, bp, color):
    artists = bp['boxes'] + bp['whiskers'] + bp['caps']
    for a in artists:
        a.set_color(color)
    for a in bp['medians']:
        a.set_color('white')


class BasePlot:

    def __init__(
        self,
        title='',
        height=3900.0,
        width=7200.0,
        dpi=600.0,
        plot3d=False,
        border=True,
    ):
        """
        ## Description

        Creates a Plot object.

        ## Arguments

        * `title`: The title of the plot. See `set_title`.
        * `height`: The height of the plot.
        * `width`: The width of the plot.
        * `dpi`: The resolution of the plot.
        * `plot3d`: Whether the plot is a 3D plot or not.
        * `border`: Whether to draw a thin black border around the frame of the plot.

        ## Example

        ~~~python
        plot = pl.Plot('Title')
        ~~~

        ![basic](../../assets/images/api/basic.png)

        """
        usetex(True, silent=True)
        self.rcparams = {
            'legend.title_fontsize': 12,
            'legend.fontsize': 14,
        }
        self.dpi = float(dpi)
        self.figure = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        self.height = height
        self.width = width
        self.title = self.figure.suptitle(title, fontsize=FONT_SIZE)
        if plot3d:
            # self.axes = self.figure.add_subplot(1, 1, 1, projection='3d')
            self.axes = self.figure.gca(projection='3d')
            self.subtitle = self.axes.text(
                x=0.5,
                y=0.90,
                z=0.0,
                s='',
                transform=self.figure.transFigure,
                fontsize=FONT_SIZE-2,
                ha='center',
                style='italic',
                color='gray',
            )
        else:
            self.axes = self.figure.add_subplot(1, 1, 1, frameon=border)
            self.subtitle = self.axes.text(
                x=0.5,
                y=0.90,
                s='',
                transform=self.figure.transFigure,
                fontsize=FONT_SIZE-2,
                ha='center',
                style='italic',
                color='gray',
            )
        self.canvas = self.axes  # for backward compatibility
        self.set_palette('maureen')
        self.set_grid('horizontal')
        self.colormap = COLORMAP_3D
        self._box_num_sets = 2
        self._box_curr_set = 0
        self.set_legend(loc='best')
        self._outset_bbox_to_anchor = {
            'upper center': (0.5, 1.12),
            'lower center': (0.5, -0.30),
            'upper right': (1.25, 1.025),
            'lower right': (1.25, -0.025),
        }
        self.markers = cycle(MARKERS)
        self.axes.tick_params(width=0.8)

    def _preprint(self):
        handles, labels = self.axes.get_legend_handles_labels()
        if len(handles) > 0:
            legend_options = copy.copy(self._legend_options)
            show = legend_options.pop('visible', True)
            legend = self.axes.legend(frameon=True, **legend_options)
            legend.set_visible(show)

    def _3d_preprocess(self, x, y, z):
        assert x.ndim == y.ndim, 'x, y shape mismatch'
        ndim = x.ndim
        if z is None:
            if ndim == 1:
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
            else:
                X, Y = x, y
                z = z + np.zeros_like(X)
        elif isinstance(z, (int, float, complex)):
            if ndim == 1:
                X, Y = np.meshgrid(x, y)
                Z = z + np.zeros_like(X)
            else:
                X, Y = x, y
                Z = z + np.zeros_like(X)
        else:  # function
            if ndim == 1:
                Z = np.zeros((len(x), len(x)))
                for i, xi in enumerate(x):
                    for j, yj in enumerate(y):
                        Z[j, i] = z(xi, yj)
                X, Y = np.meshgrid(x, y)
            else:
                raise 'Meshgrid input with function not implemented.'
        return X, Y, Z

    def _get_edge_face_color(self, fill, color):
        face = None
        edge = None
        if fill is True:
            face = LIGHT_GRAY
            fill_bool = True
        elif fill is None:
            fill_bool = False
        if color is None:
            edge = next(self.colors)
        else:
            edge = color
        if isinstance(fill, str):
            face = fill
            edge = fill
            fill_bool = True
        return edge, face, fill_bool

    def show(self):
        with plt.rc_context(self.rcparams):
            h, w = self.figure.get_figheight(), self.figure.get_figwidth()
            dpi = self.figure.get_dpi()
            self.figure.set_dpi(75.0)
            self.figure.set_figheight(3.5, forward=True)
            self.figure.set_figwidth(3.5, forward=True)
            self.figure.set_size_inches((2*3.5, 2*3.5), forward=True)
            self._preprint()
            self.figure.show()
            plt.draw()
            plt.show(self.figure)
            self.figure.set_dpi(dpi)
            self.figure.set_figheight(h, forward=True)
            self.figure.set_figwidth(w, forward=True)

    def save(self, path, bbox_inches='tight', **kwargs):
        # create path if non-existant
        plot_dir = os.path.dirname(path)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)

        # save figure
        if path[-4:] == 'html':
            try:
                from plotly.tools import mpl_to_plotly
                from plotly.offline import plot
                h, w = self.figure.get_figheight(), self.figure.get_figwidth()
                self.set_dimensions(500, 500)
                plotly_fig = mpl_to_plotly(self.figure)
                plotly_fig['layout']['width'] = '100' 
                plotly_fig['layout']['height'] = '100'
                plotly_fig['layout']['title'] = self.title.get_text()
                plotly_fig['layout']['titlefont'] = {'size': 20.0}
                plotly_fig['layout']['autosize'] = True
                plotly_fig['layout']['showlegend'] = True
                plotly_fig['layout']['xaxis1']['autorange'] = True
                plotly_fig['layout']['yaxis1']['autorange'] = True
                content = plot(plotly_fig,
                               auto_open=False,
                               include_plotlyjs=False,
                               output_type='div',
                )
                with open(path, 'w') as f:
                    f.write(content)
                self.figure.set_figheight(h, forward=True)
                self.figure.set_figwidth(w, forward=True)
            except:
                msg = 'HTML export error. Is plotly installed ?'
                raise Exception(msg)
        else:
            with plt.rc_context(self.rcparams):
                self._preprint()
                self.figure.savefig(path, bbox_inches=bbox_inches, **kwargs)

    def numpy(self):
        from PIL import Image as PILImage
        with plt.rc_context(self.rcparams):
            self._preprint()
            canvas = self.figure.canvas
            canvas.draw()
            img_np = np.array(PILImage.frombytes('RGB',
                                                 canvas.get_width_height(),
                                                 canvas.tostring_rgb()))
        return img_np[:, :, :3]

    def errorbar(self, x, y, errors=None, vertical=True, *args, **kwargs):
        """
        ## Description

        Like `plot`, but also adds error bars on the points.

        The size of the error bars is given by `errors` (a list).

        ## Arguments

        * `x`: x-coordinates of the points to plot.
        * `y`: y-coordinates of the points to plot.
        * `errors`: list of error sizes of length `len(x)`.
        * `vertical`: whether error bars are drawn vertically or horizontally. 
        * `*args`: positional arguments passed to  Matplotlib's errorbar function.
        * `**kwargs`: keyword arguments passed to Matplotlib's errorbar function.

        ## Example

        ~~~python
        x = np.arange(10)
        plot = pl.Plot('Title')
        plot.errorbar(x=x, y=x**2, errors=x, label=r'$f(x) = x^2$')
        ~~~

        ![plot](../../assets/images/api/errorbar.png)

        """
        if errors is None:
            errors = [0.0 for _ in y]
        name = 'yerr' if vertical else 'xerr'
        errors = kwargs.pop(name, errors)
        kwargs[name] = errors
        color = kwargs.pop('color', None)
        if color is None:
            color = next(self.colors)
        capthick = kwargs.pop('capthick', 2)
        capsize = kwargs.pop('capsize', 6)
        elinewidth = kwargs['linewidth'] if 'linewidth' in kwargs else 2
        elinewidth = kwargs.pop('elinewidth', elinewidth)
        marker = kwargs.pop('marker', None)
        if marker is None:
            marker = next(self.markers)
        if isinstance(marker, Marker):
            kwargs.setdefault('markerfacecolor', marker.facecolor)
            kwargs.setdefault('markeredgewidth', marker.edgewidth)
            marker = marker.symbol
        elif marker is False:
            marker = None
        markevery = kwargs.pop('markevery', None)
        if markevery is None:
            markevery = 1 if len(x) < 20 else len(x) // 10
        markersize = kwargs.pop('markersize', 8.5)
        self.axes.errorbar(
            x=x,
            y=y,
            color=color,
            capsize=capsize,
            capthick=capthick,
            elinewidth=elinewidth,
            marker=marker,
            markevery=markevery,
            markersize=markersize,
            *args,
            **kwargs,
        )

    def plot(
        self,
        x,
        y=None,
        jitter=0.000,
        smooth_window=0,
        smooth_std=True,
        *args,
        **kwargs,
    ):
        """
        ## Description

        Plots a line defined by points in `x` (and `y`).

        ## Arguments

        * `x`: x-coordinates.
        * `y`: y-coordinates, if None then `x` is used as `y` and `x` is `range(0, len(x))`.
        * `jitter`: Pointwise or float value for shading around curve.
        * `*args`: positional arguments passed to  Matplotlib's plot function.
        * `**kwargs`: keyword arguments passed to Matplotlib's plot function.

        ## Example

        ~~~python
        x = np.arange(10)
        plot = pl.Plot('Title')
        plot.plot(x=x, y=x**2, jitter=5.0, label=r'$f(x) = x^2$', linestyle='dashed')
        ~~~

        ![plot](../../assets/images/api/plot.png)

        """
        if y is None:
            y = x
            x = list( range(1, 1 +len(y)))

        if smooth_window > 0:
            smooth_x = [x[0], ]
            smooth_y = [y[0], ]
            std_y = [0.0, ]
            for i in range(smooth_window, len(x), smooth_window):
                smooth_x.append(np.mean(x[i-smooth_window: i + smooth_window]))
                smooth_y.append(np.mean(y[i-smooth_window: i + smooth_window]))
                std_y.append(np.std(y[i-smooth_window: i + smooth_window]))
            smooth_x.append(x[-1])
            smooth_y.append(y[-1])
            std_y.append(0.0)
            if smooth_std and jitter == 0.0:
                jitter = std_y
            x = smooth_x
            y = smooth_y

        color = kwargs.pop('color', None)
        if color is None:
            color = next(self.colors)
        marker = kwargs.pop('marker', None)
        if marker is None:
            marker = next(self.markers)
        if isinstance(marker, Marker):
            kwargs.setdefault('markerfacecolor', marker.facecolor)
            kwargs.setdefault('markeredgewidth', marker.edgewidth)
            marker = marker.symbol
        elif marker is False:
            marker = None
        markevery = kwargs.pop('markevery', None)
        if markevery is None:
            markevery = 1 if len(x) < 20 else len(x) // 10
        markersize = kwargs.pop('markersize', 8.5)
        self.axes.plot(
            x,
            y,
            color=color,
            marker=marker,
            markevery=markevery,
            markersize=markersize,
            *args,
            **kwargs)
        if isinstance(jitter, list):
            top = [v + j for v, j in zip(y, jitter)]
            low = [v - j for v, j in zip(y, jitter)]
            self.axes.fill_between(x, low, top, alpha=0.15, linewidth=0.0, color=color)
        elif jitter > 0.0:
            x = np.array(x)
            y = np.array(y)
            top = y + jitter
            low = y - jitter
            self.axes.fill_between(x, low, top, alpha=0.15, linewidth=0.0, color=color)

    def bar(self, x, y, show_values=False, num_box_sets=None, spacing=1.0, center_ticks=False, *args, **kwargs):

        # Arguments and Defaults
        if num_box_sets is not None:
            self._box_num_sets = num_box_sets

        if isinstance(y[0], Iterable):
            # TODO: Compute CI95
            y_means = [np.mean(ys) for ys in y]
            y_stds = [np.std(ys) for ys in y]
            sqrt_n_runs = [float(len(ys))**0.5 for ys in y]
            y_ci95 = [
                1.96 * y_std / snr
                for y_std, snr in zip(y_stds, sqrt_n_runs)
            ]
        else:
            y_means = y
            y_ci95 = [None] * len(y)
        assert len(x) == len(y_means), 'x, y not same shapes'

        color = kwargs.pop('color', None)
        if color is None:
            color = next(self.colors)
        if isinstance(color, Iterable):
          colors = color
        else:
          colors = [color, ] * len(y_means)
        label = kwargs.pop('label', None)
        assert len(x) == len(y), 'x, y not same length'
        box_set_spacing = 0.9

        # Draw bars
        positions = kwargs.pop('positions', None)
        if positions is None:
            positions = np.arange(len(x)) * (spacing * self._box_num_sets) \
                + self._box_curr_set * box_set_spacing
        for pos, value, ci95, col in zip(positions, y_means, y_ci95, colors):
            self.axes.bar(
                x=pos,
                height=value,
                width=0.7,
                color=col,
                #  edgecolor='black',
                yerr=ci95,
                capsize=7,
                *args,
                **kwargs,
            )

        if show_values:
            x_means = positions
            # Reversed because we want the latest plotted ticks.
            # (e.g. when plotted one at a time.)
            margin = 0.05 * min(y_means)
            for x_m, y_m, col in zip(reversed(x_means), reversed(y_means), reversed(colors)):
                text = self.axes.text(x_m, y_m-margin, '%.2f' % y_m, color='white',
                                        horizontalalignment='center',
                                        verticalalignment='top',
                                        fontweight='bold')
                text.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground=col)])

        # Ticks formatting
        self.axes.set_xticks(ticks=[], minor=False)
        if center_ticks:
            mid_positions = np.array(range(len(x))) * (spacing * self._box_num_sets) + (self._box_num_sets - 1) * (box_set_spacing / 2.0)
            self.axes.set_xticks(ticks=mid_positions, minor=False)
            self.axes.set_xticklabels(x)
        else:
            self.axes.set_xticks(ticks=positions, minor=False)
            self.axes.set_xticklabels(x)
        plt.setp(self.axes.get_xticklabels(),
                 rotation=35,
                 ha='right',
                 rotation_mode='anchor')
        
        # Legend
        if label is not None and not isinstance(color, Iterable):
            self.plot([], color=color, label=label, marker=False)
        self._box_curr_set += 1
        return positions


    def box(self, x, y, show_values=False, num_box_sets=None, spacing=2.0, center_ticks=False, *args, **kwargs):
        # Arguments and Defaults
        if num_box_sets is not None:
            self._box_num_sets = num_box_sets
        if not isinstance(y[0], Iterable):
            y = [y, ]
        if isinstance(x, str):
            x = [x, ]
        color = kwargs.pop('color', None)
        if color is None:
            color = next(self.colors)
        label = kwargs.pop('label', None)
        assert len(x) == len(y), 'x, y not same length'

        # Boxes
        positions = kwargs.pop('positions', None)
        if positions is None:
            positions = np.array(range(len(x))) * (spacing * self._box_num_sets) \
                      + self._box_curr_set * 0.8
        boxplot = self.axes.boxplot(y,
                                      positions=positions,
                                      sym='',
                                      patch_artist=True,
                                      widths=0.6,
                                      labels=x,
                                      )
        set_box_color(self.axes, boxplot, color)

        # Print median values on plot
        if show_values:
            y_means = [stats.median(v) for v in y]
            x_means = self.axes.get_xticks()
            # Reversed because we want the latest plotted ticks.
            # (e.g. when plotted one at a time.)
            for x_m, y_m in zip(reversed(x_means), reversed(y_means)):
                text = self.axes.text(x_m, y_m, '%.2f' % y_m, color='white',
                                        horizontalalignment='center',
                                        verticalalignment='bottom',
                                        fontweight='bold')
                text.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground=color)])

        # Ticks formatting
        if center_ticks:
            mid_positions = np.array(range(len(x))) * (spacing * self._box_num_sets) + (self._box_num_sets - 1) * 0.4
            self.axes.set_xticks(ticks=[], minor=False)
            self.axes.set_xticks(ticks=mid_positions, minor=False)
            self.axes.set_xticklabels(x)
        plt.setp(self.axes.get_xticklabels(),
                 rotation=35,
                 ha='right',
                 rotation_mode='anchor')
        
        # Legend
        if label is not None:
            self.plot([], color=color, label=label, marker=False)
        self._box_curr_set += 1
        return positions

    def scatter(self, *args, **kwargs):
        """
        ## Description

        Scatter points defined by `x` (and `y`).

        ## Arguments

        * `*args`: positional arguments passed to  Matplotlib's plot function.
        * `**kwargs`: keyword arguments passed to Matplotlib's plot function.

        ## Example

        ~~~python
        x = np.arange(10)
        plot = pl.Plot('Title')
        plot.scatter(x, x**2, label=r'$f(x) = x^2$')
        ~~~

        ![plot](../../assets/images/api/scatter.png)

        """
        color = kwargs.pop('color', None)
        if color is None:
            color = next(self.colors)
        marker = kwargs.pop('marker', None)
        if marker is None:
            marker = next(self.markers)
        if isinstance(marker, Marker):
            marker = marker.symbol
        elif marker is False:
            marker = None
        self.axes.scatter(color=color, marker=marker, *args, **kwargs)

    def heatmap(
        self,
        heatvalues,
        xlabels=None,
        ylabels=None,
        show_values=False,
        cbar_title='',
        *args,
        **kwargs,
    ):
        """
        ## Description

        Draws a heatmap given an array of heat values.

        ## Arguments

        * `heatvalues`: 2D grid to plot. (list of list or np.array)
        * `xlabels`: list of names.
        * `ylabels`: list of names.
        * `show_values`: bool of whether to write values inside heat box.
        * `cbar_title`: title of the color bar.
        * `*args`: positional arguments passed to  Matplotlib's imshow function.
        * `**kwargs`: keyword arguments passed to Matplotlib's imshow function.

        Keyword arguments of interest:

        * `interpolation`: nearest / kaiser / hanning / gaussian / spline16

        ## Example

        ~~~python
        values = np.arange(100).reshape(10, 10)
        plot = pl.Plot(height=3900.0, width=7200.0)
        plot.heatmap(
            heatvalues=values,
            xlabels=[str(x) for x in range(10)],
            ylabels=[str(10 - x) for x in range(10)],
            show_values=True,
            cbar_title='My Color Bar',
        )
        ~~~

        ![plot](../../assets/images/api/heatmap.png)

        """
        self.set_grid('none')
        self.axes.tick_params(axis=u'both', which=u'both',length=0)
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        cmap = kwargs.pop('cmap', None)
        interpolation = kwargs.pop('interpolation', 'nearest')
        aspect = kwargs.pop('aspect', None)
        origin = kwargs.pop('origin', 'upper')
        norm = kwargs.pop('norm', mpl.colors.Normalize())
        if 'maureen' not in self.palette:
            cmap = self.palette
        im = self.axes.imshow(heatvalues,
                                cmap=cmap,
                                norm=norm,
                                interpolation=interpolation,
                                aspect=aspect,
                                origin=origin,
                                *args,
                                **kwargs)
        cbar = self.figure.colorbar(im, ax=self.axes)
        cbar.ax.set_ylabel(cbar_title, rotation=-90, va="bottom")
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(axis=u'both', which=u'both', length=0)
        if xlabels is not None:
            self.axes.set_xticks(np.arange(len(heatvalues[0])))
            self.axes.set_xticklabels(xlabels)
            plt.setp(self.axes.get_xticklabels(),
                     rotation=35,
                     ha='right',
                     rotation_mode='anchor')

        if ylabels is not None:
            self.axes.set_yticks(np.arange(len(heatvalues)))
            self.axes.set_yticklabels(ylabels)

        if show_values:
            for y, row in enumerate(heatvalues):
                for x, val in enumerate(row):
                    color = 'white' if im.norm(val) > 0.5 else 'black'
                    im.axes.text(x, y, '%.2f' % val, color=color,
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontweight='light')
        # Recenter the plot because of colorbar
        self.figure.subplots_adjust(left=0.3, right=0.8)

    def contour(self, x, y, z=None, fill=True, *args, **kwargs):
        """
        ## Description

        Draws the contours of a 3D function.

        ## Arguments

        * `x`: x-coordinates of the points where z is evaluated.
        * `y`: y-coordinates of the points where z is evaluated.
        * `z`: function or 2D-array of values where a function was evaluated given `x` and `y` values.
        * `fill`: whether to fill the contour plot.
        * `*args`: positional arguments passed to  Matplotlib's contour / contourf function.
        * `**kwargs`: keyword arguments passed to Matplotlib's contour / contourf function.

        ## Example

        ~~~python
        x = y = np.linspace(-10, 10, num=100)
        f = lambda x, y: 0.5 * x**3 + 10.0 * y**2
        plot = pl.Plot('Title')
        plot.contour(x, y, f, fill=True)
        ~~~

        ![plot](../../assets/images/api/contour.png)

        """
        X, Y, Z = self._3d_preprocess(x, y, z)
        self.set_grid(axis=None)
        if fill:
            cont = self.canvas.contourf(
                X, Y, Z,
                zdir='x',
                cmap=self.colormap,
                *args,
                **kwargs,
            )
        else:
            cont = self.axes.contour(X, Y, Z, zdir='x', cmap=self.colormap, *args, **kwargs)
        return cont

    def circle(self, x, y, radius, fill=None, color=None, alpha=0.5,
               linewidth=1.5, *args, **kwargs):
        edgecolor, facecolor, fill_bool = self._get_edge_face_color(fill, color)
        c = Circle((x, y), radius, edgecolor=edgecolor, facecolor=facecolor,
                   alpha=alpha, linewidth=linewidth, fill=fill_bool, *args, **kwargs)
        self.axes.add_artist(c)

    def rectangle(self, x, y, width, height, fill=None, color=None, alpha=0.5,
                  linewidth=1.5, *args, **kwargs):
        edgecolor, facecolor, fill_bool = self._get_edge_face_color(fill, color)
        x = x - width / 2.0
        y = y - height / 2.0
        r = Rectangle((x, y), width=width, height=height, edgecolor=edgecolor,
                      facecolor=facecolor, alpha=alpha, linewidth=linewidth,
                      fill=fill_bool, *args, **kwargs)
        self.axes.add_artist(r)

    def arrow(self, start, end, width=1.0, fill=None, color=None, linewidth=1.0,
              *args, **kwargs):
        """
        Start and end are (z, y) tuples.
        """
        edgecolor, facecolor, fill_bool = self._get_edge_face_color(fill, color)
        x = start[0]
        y = start[1]
        dx = end[0] - x
        dy = end[1] - y
        r = Arrow(x, y, dx, dy, width=width, edgecolor=edgecolor,
                  facecolor=facecolor, fill=fill_bool, linewidth=linewidth,
                  *args, **kwargs)
        self.axes.add_artist(r)

    def fancybox(self, x, y, width, height, style='round', fill=None,
                 color=None, alpha=0.5, linewidth=1.5, *args, **kwargs):
        edgecolor, facecolor, fill_bool = self._get_edge_face_color(fill, color)
        default_pad = 0.3
        width = width - default_pad * 2.0
        height = height - default_pad * 2.0
        x = x - width / 2.0
        y = y - height / 2.0
        styles = {'circle': BoxStyle.Circle, 'darrow': BoxStyle.DArrow, 'larrow': BoxStyle.LArrow,
                  'rarrow': BoxStyle.RArrow, 'round': BoxStyle.Round, 'round4': BoxStyle.Round4,
                  'roundtooth': BoxStyle.Roundtooth, 'sawtooth': BoxStyle.Sawtooth,
                  'square': BoxStyle.Square, }
        sty_args = {
                'circle': {'pad': default_pad, },
                'darrow': {'pad': default_pad, },
                'larrow': {'pad': default_pad, },
                'rarrow': {'pad': default_pad, },
                'round': {'pad': default_pad, 'rounding_size': None},
                'round4': {'pad': default_pad, 'rounding_size': None},
                'roundtooth': {'pad': default_pad, 'tooth_size': None},
                'sawtooth': {'pad': default_pad, 'tooth_size': None},
                'square': {'pad': default_pad, },
                }
        style = styles[style](**sty_args[style])
        r = FancyBboxPatch((x, y), width=width, height=height, boxstyle=style, edgecolor=edgecolor,
                      facecolor=facecolor, alpha=alpha, linewidth=linewidth,
                      fill=fill_bool, *args, **kwargs)
        self.axes.add_artist(r)

    def text(self, text, xytext, *args, **kwargs):
        self.axes.annotate(text, xytext, textcoords='data', *args, **kwargs)

    def annotate(self, text, xytext, xylabel, rad=0.0, shape='->', width=0.5,
                 color='#000000', *args, **kwargs):
        arrowprops = {
                'arrowstyle': shape,
                'connectionstyle': 'arc3, rad=' + str(rad),
                # 'connectionstyle': 'arc3',
                'lw': width,
                'facecolor': color,
                'edgecolor': color,
                }
        self.axes.annotate(text, xylabel, xytext, xycoords='data', textcoords='data', 
                arrowprops=arrowprops, *args, **kwargs)

    def set_font(self, name):
        self.update_rcparams({
            'mathtext.fontset': 'cm',
            'font.family': name,
            'font.sans-serif': name,
            'font.serif': name,
        })

    def set_title(
        self,
        title,
        loc='center',
        x=None,
        y=0.98,
        text_obj=None,
        **kwargs,
    ):
        """
        ## Description

        Sets the title of the plot.

        ## Arguments

        * `title`: Text of the title.
        * `loc`: Location of the title.
        * `x`: x-coordinate of the title.
        * `y`: y-coordinate of the title.
        * `font`: Optional font name string.
        * `text_obj`: A text object where the title is set.

        ## Example

        ~~~python
        plot = pl.Plot('Title')
        plot.set_title('New and Much Longer Title', loc='right', x=0.9, y=0.92)
        ~~~

        ![plot](../../assets/images/api/set_title.png)

        """
        if text_obj is None:
            text_obj = self.title
        text_obj.set_text(title)
        if x is not None:
            text_obj.set_x(x)
        text_obj.set_y(y)
        if loc == 'center':
            if x is None:
                text_obj.set_x(0.5)
            text_obj.set_ha('center')
        if loc == 'left':
            if x is None:
                text_obj.set_x(0.1)
            text_obj.set_ha('left')
        if loc == 'right':
            if x is None:
                text_obj.set_x(0.9)
            text_obj.set_ha('right')

    def set_subtitle(self, *args, **kwargs):
        if 'y' not in kwargs:
            kwargs['y'] = 0.9
        self.set_title(text_obj=self.subtitle, *args, **kwargs)

    def set_notation(self, x=None, y=None):
        if x == 'scientific':
            xra = 0.0
        elif x == 'decimal':
            xra = float('inf')
        if y == 'scientific':
            yra = 0.0
        elif y == 'decimal':
            yra = float('inf')
        if x is not None:
            self.axes.ticklabel_format(style='sci', axis='x', scilimits=(-xra, xra))
        if y is not None:
            self.axes.ticklabel_format(style='sci', axis='y', scilimits=(-yra, yra))

    def set_axis(self, xtitle='', ytitle=''):
        self.axes.set_xlabel(xtitle)
        self.axes.set_ylabel(ytitle)

    def set_palette(self, palette, num_colors=8):
        self.palette = palette
        if 'maureen' in palette or 'custom' in palette:
            palette = MAUREENSTONE_COLORS
        elif palette == 'vibrant':
            vibrant_order = ['cyan', 'magenta', 'orange',
                             'teal', 'blue', 'red', 'gray']
            palette = [Vibrant[vo] for vo in vibrant_order]
        elif isinstance(palette, str):
            cmap = mpl.cm.get_cmap(palette)
            palette = [cmap(i) for i in np.linspace(0.1, 1.0, num_colors)]
        self.color_list = palette
        self.colors = cycle(self.color_list)

    def set_colormap(self, cm):
        self.colormap = cm

    def set_xticks(self, positions, labels=None):
        self.axes.set_xticks([])
        self.axes.set_xticks(positions)
        if labels is not None:
            self.axes.set_xticklabels(labels)

    def set_yticks(self, positions, labels=None):
        self.axes.set_yticks([])
        self.axes.set_yticks(positions)
        if labels is not None:
            self.axes.set_yticklabels(labels)

    def set_dimensions(self, height=None, width=None):
        if height is not None:
            self.height = height
            self.figure.set_figheight(height/self.dpi, forward=True)
        if width is not None:
            self.width = width
            self.figure.set_figwidth(width/self.dpi, forward=True)

    def set_dpi(self, dpi):
        self.figure.set_dpi(dpi)
        self.dpi = float(dpi)

    def set_scales(self, x=None, y=None, z=None):
        """
        Possible values: 'linear', 'log', 'log2', 'symlog', 'symlog2', 'logit'
        """
        def scale_base(scale):
            return (scale[:-1], 2) if '2' in scale else (scale, 10)
        if x is not None:
            x_scale, x_base = scale_base(x)
            if x_scale in ('log', 'symlog'):
                self.axes.set_xscale(x_scale, base=x_base)
            else:
                self.axes.set_xscale(x_scale)
        if y is not None:
            y_scale, y_base = scale_base(y)
            if y_scale in ('log', 'symlog'):
                self.axes.set_yscale(y_scale, base=y_base)
            else:
                self.axes.set_yscale(y_scale)
        if z is not None:
            z_scale, z_base = scale_base(z)
            if z_scale in ('log', 'symlog'):
                self.axes.set_zscale(z_scale, base=z_base)
            else:
                self.axes.set_zscale(z_scale)

    def set_lims(self, x=None, y=None, z=None):
        """
        Expects (min_val, max_val) tuples for each arguments, where vals can be
        none to re-use current ones.
        """
        if x is not None:
            xmin, xmax = self.axes.get_xlim()
            if x[0] is not None:
                xmin = x[0]
            if x[1] is not None:
                xmax = x[1]
            self.axes.set_xlim(xmin, xmax)
        if y is not None:
            ymin, ymax = self.axes.get_ylim()
            if y[0] is not None:
                ymin = y[0]
            if y[1] is not None:
                ymax = y[1]
            self.axes.set_ylim(ymin, ymax)
        if z is not None:
            zmin, zmax = self.axes.get_zlim()
            if z[0] is not None:
                zmin = z[0]
            if z[1] is not None:
                zmax = z[1]
            self.axes.set_zlim(zmin, zmax)

    def set_grid(self, axis='full', granularity='fine'):
        """
        Sets a fine, light gray background grid.

        axis values: full, vertical, horizontal, none.
        granularity: coarse, fine
        """
        both = axis == 'full'
        if granularity == 'fine':
            granularity = 'both'
        elif granularity == 'coarse':
            granularity = 'major'
        if both or axis == 'vertical':
            self.axes.xaxis.grid(
                which=granularity,
                color=LIGHT_GRAY,
                linestyle='-',
                linewidth=0.7,
            )
        if both or axis == 'horizontal':
            self.axes.yaxis.grid(
                which=granularity,
                color=LIGHT_GRAY,
                linestyle='-',
                linewidth=0.7,
            )
        if axis is None or axis == 'none' or axis == 'off':
            self.axes.xaxis.grid(False)
            self.axes.yaxis.grid(False)

    def set_legend(
        self,
        loc='best',
        title=None,
        show=True,
        inset=True,
        ncol=1,
        alpha=0.8,
        round_corners=False,
        **kwargs,
    ):
        # Here process the position
        legend_options = {}
        legend_location = loc
        bbox_to_anchor = None
        if not inset:
            bbox_to_anchor = self._outset_bbox_to_anchor[loc]

        legend_options = {
            'loc': legend_location,
            'bbox_to_anchor': bbox_to_anchor,
            'visible': show,
            'ncol': ncol,
            'framealpha': alpha,
            'title': title,
            'fancybox': round_corners,
            'title_fontproperties': {
                'weight': 'bold',
            }
        }
        legend_options.update(kwargs)
        self._legend_options = legend_options

    def update_rcparams(self, updates):
        self.rcparams.update(updates)

    def save_legend(self, path):
        handles, labels = self.axes.get_legend_handles_labels()
        if len(handles) > 0:
            legend_options = copy.copy(self._legend_options)
            show = legend_options.pop('visible', True)
            legend = self.axes.legend(frameon=True, **legend_options)
            expand = [-5, -5, 5, 5]
            fig = legend.figure
            fig.canvas.draw()
            bbox = legend.get_window_extent()
            bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(path, dpi="figure", bbox_inches=bbox)

    def stretch(self, left=0.0, right=0.0, top=0.0, bottom=0.0):
        self.figure.subplots_adjust(left=0.125 + left,
                                    right=0.9 + right,
                                    top=0.9 + top,
                                    bottom=0.1 + bottom)

    def close(self):
        plt.close(self.figure)

    def usetex(self, *args, **kwargs):
        usetex(*args, **kwargs)


class Drawing(BasePlot):

    def __init__(self, *args, **kwargs):
        super(Drawing, self).__init__(*args, **kwargs)
        self.axes.axis('off')

    def _preprint(self, *args, **kwargs):
        super(Drawing, self)._preprint(*args, **kwargs)
        if self.axes.legend_ is not None:
            self.axes.legend_.remove()


class Image(Drawing):

    def __init__(self, path, *args, **kwargs):
        super(Image, self).__init__(*args, **kwargs)
        image = mpimg.imread(path)
        self.axes.imshow(image)


class Plot3D(BasePlot):

    def __init__(self, *args, **kwargs):
        super(Plot3D, self).__init__(plot3d=True, *args, **kwargs)

    def plot(self, x, y, z=0, jitter=None, *args, **kwargs):
        if hasattr(z, '__call__'):
            z = np.array([z(x1, y1) for x1, y1 in zip(x, y)])
        color = kwargs.pop('color', None)
        if color is None:
            color = next(self.colors)
        self.axes.plot(x, y, zs=z, color=color, *args, **kwargs)

    def scatter(self, x, y, z=0, *args, **kwargs):
        if hasattr(z, '__call__'):
            z = np.array([z(x1, y1) for x1, y1 in zip(x, y)])
        color = kwargs.pop('color', None)
        if color is None:
            color = next(self.colors)
        self.axes.scatter(xs=x, ys=y, zs=z, c=color, *args, **kwargs)

    def surface(self, x, y, z=None, alpha=0.25, linewidth=0, *args, **kwargs):
        X, Y, Z = self._3d_preprocess(x, y, z)
        self.axes.plot_surface(X=X, Y=Y, Z=Z, rstride=1, cstride=1,
                                 cmap=self.colormap,
                                 linewidth=linewidth,
                                 antialiased=True,
                                 alpha=alpha,
                                 *args,
                                 **kwargs)

    def wireframe(self, x, y, z=None, *args, **kwargs):
        X, Y, Z = self._3d_preprocess(x, y, z)
        self.axes.plot_wireframe(X=X, Y=Y, Z=Z, color=next(self.colors), *args, **kwargs)

    def projection(self, x, y, z=None, alpha=0.1, linewidth=0, *args, **kwargs):
        X, Y, Z = self._3d_preprocess(x, y, z)
        xmin = np.min(x)
        ymin = np.min(Y)
        zmin = np.min(Z)
        self.axes.plot_surface(X, Y, Z, rstride=1, cstride=1,
                linewidth=linewidth, alpha=alpha, cmap=self.colormap,
                *args, **kwargs)
        self.axes.contour(X, Y, Z, zdir='z', offset=zmin, cmap=self.colormap)
        self.axes.contour(X, Y, Z, zdir='x', offset=xmin, cmap=self.colormap)
        self.axes.contour(X, Y, Z, zdir='y', offset=ymin, cmap=self.colormap)

    def set_camera(self, elev=None, azim=None):
        """
        Both parameters are angles in [0, 360].
        """
        self.axes.view_init(elev, azim)

    def set_notation(self, x=None, y=None, z=None):
        if x == 'scientific':
            xra = 0.0
        elif x == 'decimal':
            xra = float('inf')
        if y == 'scientific':
            yra = 0.0
        elif y == 'decimal':
            yra = float('inf')
        if z == 'scientific':
            zra = 0.0
        elif z == 'decimal':
            zra = float('inf')
        print(x, y, z)
        if x is not None:
            self.axes.ticklabel_format(style='sci', axis='x', scilimits=(-xra, xra))
        if y is not None:
            self.axes.ticklabel_format(style='sci', axis='y', scilimits=(-yra, yra))
        if z is not None:
            self.axes.ticklabel_format(style='sci', axis='z', scilimits=(-zra, zra))

    def set_axis(self, xtitle='', ytitle='', ztitle='', notation='scientific'):
        self.axes.set_xlabel(xtitle, labelpad=25)
        self.axes.set_ylabel(ytitle, labelpad=25)
        self.axes.set_zlabel(ztitle, labelpad=25)


class Container(BasePlot):

    def __init__(self, rows=1, cols=2, height=None, width=None, *args, **kwargs):
        if height is None:
            height = 2600.0 * rows
        if width is None:
            width = 4200.0 * cols
        super(Container, self).__init__(height=height, width=width, *args, **kwargs)
        self.axes.axis('off')
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)
        self.axeses = []
        self.plots = []
        for i in range(rows):
            self.plots.append([])
            self.axeses.append([])
            for j in range(cols):
                subplot = self.figure.add_subplot(rows, cols, i * cols + j + 1)
                subplot.axis('off')
                self.plots[i].append(None)
                self.axeses[i].append(subplot)
#        self.figure.set_tight_layout(False)

    def _preprint(self):
        for i in range(self.rows):
            for j in range(self.cols):
                plot = self.plots[i][j]
                canvas = self.axeses[i][j]
                if plot is not None:
                    img_fname = TEMP_FILENAME + str(time()) + TEMP_FILE_EXT
                    plot.save(img_fname, bbox_inches='tight')
                    image = mpimg.imread(img_fname)
                    canvas.imshow(image)
                    os.remove(img_fname)
        super(Container, self)._preprint()

    def set_plot(self, row, col, plot):
        self.plots[row][col] = plot

    def get_plot(self, row, col):
        return self.plots[row][col]


class Animation(object):

    def __init__(self, fps=24):
        self.fps = fps
        self.reset()

    def reset(self):
        self.frames = []
        self._iter = -1

    def add_frame(self, image):
        """
        image should be a (height, width, 3) np.ndarry
        """
        self.frames.append(np.copy(image))

    def anim_fn(self, fn, data):
        """
        fn: a function that returns a plot
        data: an iterable
        """
        for i in range(len(data)):
            p = fn(data[:i])
            self.add_frame(p.numpy())

    def rotate_3d(self, plot, duration=8):
        nframes = duration * self.fps
        change_angle = 360.0 / nframes
        azim = plot.canvas.azim
        for i in range(nframes):
            plot.set_camera(azim=azim + i * change_angle)
            self.frames.append(plot.numpy())

    def _make(self, t):
        self._iter += 1
        # Weird bug where idx might be = len(self.frames)
        idx = min(self._iter, len(self.frames) - 1)
        return self.frames[idx]

    def save(self, path):
        from moviepy.editor import VideoClip
        fname, ext = os.path.splitext(path)
        duration = (len(self.frames) - 1) / float(self.fps)
        self.animation = VideoClip(self._make, duration=duration)
        if 'gif' in ext:
            self.animation.write_gif(path, fps=self.fps)
        else:
            self.animation.write_videofile(path, fps=self.fps)
        self._iter = -1
