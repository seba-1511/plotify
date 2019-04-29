#!/usr/bin/env python3

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #required for 3D plots
from matplotlib import image as mpimg
from matplotlib.patches import Circle, Rectangle, Arrow, FancyBboxPatch, BoxStyle

from tempfile import gettempdir
from collections import Iterable
from itertools import cycle
from time import time


try:
    from io import BytesIO as BytesIO
except:
    pass

MEM_IMG = BytesIO()


FONT_SIZE = 20
MAUREENSTONE_COLORS = ['#396AB1', '#DA7C30', '#3E9651', '#CF2529', '#535154', '#6B4C9A', '#922428', '#948B3D']
Maureen = {
    'blue': MAUREENSTONE_COLORS[0],
    'orange': MAUREENSTONE_COLORS[1],
    'green': MAUREENSTONE_COLORS[2],
    'red': MAUREENSTONE_COLORS[3],
    'black': MAUREENSTONE_COLORS[4],
    'purple': MAUREENSTONE_COLORS[5],
    'cardinal': MAUREENSTONE_COLORS[6],
    'gold': MAUREENSTONE_COLORS[7],

}
LIGHT_GRAY = '#D3D3D3'
PALETTE_NAME = MAUREENSTONE_COLORS
COLORMAP_3D = 'YlGnBu'
TEMP_FILENAME = os.path.join(gettempdir(), 'sebplot')
TEMP_FILE_EXT = '.png'

mpl.style.use('seaborn-poster')

# Latex font
#mpl.rc('text', usetex=True)  # Renders math with latex program (slow)
mpl.rcParams['mathtext.fontset'] = 'cm'  # Font for tex


class Plot(object):

    def __init__(self, title='', height=3900.0, width=7200.0, dpi=600.0, plot3d=False, border=True):
        self.dpi = float(dpi)
        self.figure = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        self.height = height
        self.width = width
        self.title = self.figure.suptitle(title, fontsize=FONT_SIZE)
        if plot3d:
            # self.canvas = self.figure.add_subplot(1, 1, 1, projection='3d')
            self.canvas = self.figure.gca(projection='3d')
            self.subtitle = self.canvas.text(x=0.5, y=0.90, z=0.0, s='', transform=self.figure.transFigure, fontsize=FONT_SIZE-2, ha='center', style='italic', color='gray')
        else:
            self.canvas = self.figure.add_subplot(1, 1, 1, frameon=border)
            self.subtitle = self.canvas.text(x=0.5, y=0.90, s='', transform=self.figure.transFigure, fontsize=FONT_SIZE-2, ha='center', style='italic', color='gray')
        self.set_palette('maureen')
        self.set_grid('horizontal')
        self.colormap = COLORMAP_3D
#        self.legend_location = 'upper right'
        self.legend_location = 'best'

    def _preprint(self):
        handles, labels = self.canvas.get_legend_handles_labels()
        if len(handles) > 0:
            l = self.canvas.legend(frameon=True, loc=self.legend_location)

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

    def save(self, path, **kwargs):
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
            self._preprint()
            self.figure.savefig(path, **kwargs)

    def numpy(self):
        from PIL import Image as PILImage
        self._preprint()
        canvas = self.figure.canvas
        canvas.draw()
        img_np = np.array(PILImage.frombytes('RGB',
                                             canvas.get_width_height(),
                                             canvas.tostring_rgb()))
        return img_np[:, :, :3]

    def errorbar(self, x, y, errors=None, vertical=True, *args, **kwargs):
        if errors is None:
            errors = [0.0 for _ in y]
        name = 'yerr' if vertical else 'xerr'
        errors = kwargs.pop(name, errors)
        kwargs[name] = errors
        color = kwargs.pop('color', next(self.colors))
        capthick = kwargs.pop('capthick', 2)
        capsize = kwargs.pop('capsize', 6)
        elinewidth = kwargs['linewidth'] if 'linewidth' in kwargs else 2
        elinewidth = kwargs.pop('elinewidth', elinewidth)
        self.canvas.errorbar(x=x, y=y, color=color, capsize=capsize, capthick=capthick,
                             elinewidth=elinewidth, *args, **kwargs)

    def plot(self, x, y=None, jitter=0.000, smooth_window=0, smooth_std=True, *args, **kwargs):
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

        color = kwargs.pop('color', next(self.colors))
        self.canvas.plot(x, y, color=color, *args, **kwargs)
        if isinstance(jitter, list):
            top = [v + j for v, j in zip(y, jitter)]
            low = [v - j for v, j in zip(y, jitter)]
            self.canvas.fill_between(x, low, top, alpha=0.15, linewidth=0.0, color=color)
        elif jitter > 0.0:
            x = np.array(x)
            y = np.array(y)
            top = y + jitter
            low = y - jitter
            self.canvas.fill_between(x, low, top, alpha=0.15, linewidth=0.0, color=color)

    def bar(self, x, y=None, *args, **kwargs):
        # TODO: Show variance and mean on bars (as an option)
        if y is None:
            y = x
            x = list(range(len(y)))
        fake_x = list(range(len(x)))
        for i in fake_x:
            color = next(self.colors)
            self.canvas.bar(x=i, height=y[i], color=color, *args, **kwargs)
        self.canvas.set_xticks(fake_x)
        self.canvas.set_xticklabels(x)
        plt.setp(self.canvas.get_xticklabels(),
                 rotation=35,
                 ha='right',
                 rotation_mode='anchor')

    def scatter(self, *args, **kwargs):
        c = kwargs.pop('color', next(self.colors))
        self.canvas.scatter(color=c, *args, **kwargs)

    def heatmap(self, heatvalues, xlabels=None, ylabels=None, show_values=False, cbar_title='', *args, **kwargs):
        '''
        heatvalues: 2D grid to plot. (list of list or np.array)
        xlabels: list of names
        ylabels: list of names
        show_values: bool of whether to write values inside heat box
        interpolation: nearest / kaiser / hanning / gaussian / spline16
        '''
        self.set_grid('none')
        self.canvas.tick_params(axis=u'both', which=u'both',length=0)
        self.canvas.set_xticks([])
        self.canvas.set_yticks([])
        cmap = kwargs.pop('cmap', None)
        interpolation = kwargs.pop('interpolation', 'nearest')
        aspect = kwargs.pop('aspect', None)
        origin = kwargs.pop('origin', 'upper')
        norm = kwargs.pop('norm', mpl.colors.Normalize())
        if 'maureen' not in self.palette:
            cmap = self.palette
        im = self.canvas.imshow(heatvalues,
                                cmap=cmap,
                                norm=norm,
                                interpolation=interpolation,
                                aspect=aspect,
                                origin=origin,
                                *args,
                                **kwargs)
        cbar = self.figure.colorbar(im, ax=self.canvas)
        cbar.ax.set_ylabel(cbar_title, rotation=-90, va="bottom")
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(axis=u'both', which=u'both', length=0)
        if xlabels is not None:
            self.canvas.set_xticks(np.arange(len(heatvalues[0])))
            self.canvas.set_xticklabels(xlabels)
            plt.setp(self.canvas.get_xticklabels(),
                     rotation=35,
                     ha='right',
                     rotation_mode='anchor')

        if ylabels is not None:
            self.canvas.set_yticks(np.arange(len(heatvalues)))
            self.canvas.set_yticklabels(ylabels)

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
        X, Y, Z = self._3d_preprocess(x, y, z)
        self.set_grid(axis=None)
        if fill:
            self.canvas.contourf(X, Y, Z, zdir='x', cmap=self.colormap, *args, **kwargs)
        else:
            self.canvas.contour(X, Y, Z, zdir='x', cmap=self.colormap, *args, **kwargs)

    def circle(self, x, y, radius, fill=None, color=None, alpha=0.5,
               linewidth=1.5, *args, **kwargs):
        edgecolor, facecolor, fill_bool = self._get_edge_face_color(fill, color)
        c = Circle((x, y), radius, edgecolor=edgecolor, facecolor=facecolor,
                   alpha=alpha, linewidth=linewidth, fill=fill_bool, *args, **kwargs)
        self.canvas.add_artist(c)

    def rectangle(self, x, y, width, height, fill=None, color=None, alpha=0.5,
                  linewidth=1.5, *args, **kwargs):
        edgecolor, facecolor, fill_bool = self._get_edge_face_color(fill, color)
        x = x - width / 2.0
        y = y - height / 2.0
        r = Rectangle((x, y), width=width, height=height, edgecolor=edgecolor,
                      facecolor=facecolor, alpha=alpha, linewidth=linewidth,
                      fill=fill_bool, *args, **kwargs)
        self.canvas.add_artist(r)

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
        self.canvas.add_artist(r)

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
        self.canvas.add_artist(r)

    def text(self, text, xytext, *args, **kwargs):
        self.canvas.annotate(text, xytext, textcoords='data', *args, **kwargs)

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
        self.canvas.annotate(text, xylabel, xytext, xycoords='data', textcoords='data', 
                arrowprops=arrowprops, *args, **kwargs)

    def set_title(self, title, loc='center', x=None, y=0.98, text_obj=None):
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

    def set_axis(self, xtitle='', ytitle=''):
        self.canvas.set_xlabel(xtitle)
        self.canvas.set_ylabel(ytitle)

    def set_palette(self, palette, num_colors=8):
        assert isinstance(palette, str), 'list palettes unsupported. (yet)'
        self.palette = palette
        if 'maureen' in palette or 'custom' in palette:
            palette = MAUREENSTONE_COLORS
        elif isinstance(palette, str):
            cmap = mpl.cm.get_cmap(palette)
            palette = [cmap(i) for i in np.linspace(0.1, 1.0, num_colors)]
        self.color_list = palette
        self.colors = cycle(self.color_list)

    def set_colormap(self, cm):
        self.colormap = cm

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
        Possible values: 'linear', 'log', 'symlog', 'logit'
        """
        if x is not None:
            self.canvas.set_xscale(x)
        if y is not None:
            self.canvas.set_yscale(y)
        if z is not None:
            self.canvas.set_zscale(z)

    def set_lims(self, x=None, y=None, z=None):
        """
        Expects (min_val, max_val) tuples for each arguments, where vals can be
        none to re-use current ones.
        """
        if x is not None:
            xmin, xmax = self.canvas.get_xlim()
            if x[0] is not None:
                xmin = x[0]
            if x[1] is not None:
                xmax = x[1]
            self.canvas.set_xlim(xmin, xmax)
        if y is not None:
            ymin, ymax = self.canvas.get_ylim()
            if y[0] is not None:
                ymin = y[0]
            if y[1] is not None:
                ymax = y[1]
            self.canvas.set_ylim(ymin, ymax)
        if z is not None:
            zmin, zmaz = self.canvas.get_zlim()
            if z[0] is not None:
                zmin = z[0]
            if z[1] is not None:
                zmax = z[1]
            self.canvas.set_zlim(zmin, zmax)

    def set_grid(self, axis='full'):
        """
        Sets a fine, light gray background grid.

        axis values: full, vertical, horizontal, none.
        """
        both = axis == 'full'
        if both or axis == 'vertical':
            self.canvas.xaxis.grid(
                which='major',
                color=LIGHT_GRAY,
                linestyle='-',
                linewidth=0.7,
            )
        if both or axis == 'horizontal':
            self.canvas.yaxis.grid(
                which='major',
                color=LIGHT_GRAY,
                linestyle='-',
                linewidth=0.7,
            )
        if axis is None or axis == 'none' or axis == 'off':
            self.canvas.xaxis.grid(False)
            self.canvas.yaxis.grid(False)

    def stretch(self, left=0.0, right=0.0, top=0.0, bottom=0.0):
        self.figure.subplots_adjust(left=0.125 + left,
                                    right=0.9 + right,
                                    top=0.9 + top,
                                    bottom=0.1 + bottom)

    def close(self):
        plt.close(self.figure)


class Drawing(Plot):

    def __init__(self, *args, **kwargs):
        super(Drawing, self).__init__(*args, **kwargs)
        self.canvas.axis('off')

    def _preprint(self, *args, **kwargs):
        super(Drawing, self)._preprint(*args, **kwargs)
        if self.canvas.legend_ is not None:
            self.canvas.legend_.remove()


class Image(Drawing):

    def __init__(self, path, *args, **kwargs):
        super(Image, self).__init__(*args, **kwargs)
        image = mpimg.imread(path)
        self.canvas.imshow(image)



class Plot3D(Plot):

    def __init__(self, *args, **kwargs):
        super(Plot3D, self).__init__(plot3d=True, *args, **kwargs)

    def plot(self, x, y, z=0, jitter=None, *args, **kwargs):
        if hasattr(z, '__call__'):
            z = np.array([z(x1, y1) for x1, y1 in zip(x, y)])
        color = kwargs.pop('color', next(self.colors))
        self.canvas.plot(x, y, zs=z, color=color, *args, **kwargs)

    def scatter(self, x, y, z=0, *args, **kwargs):
        if hasattr(z, '__call__'):
            z = np.array([z(x1, y1) for x1, y1 in zip(x, y)])
        color = kwargs.pop('color', next(self.colors))
        self.canvas.scatter(xs=x, ys=y, zs=z, c=color, *args, **kwargs)

    def surface(self, x, y, z=None, alpha=0.25, linewidth=0, *args, **kwargs):
        X, Y, Z = self._3d_preprocess(x, y, z)
        self.canvas.plot_surface(X=X, Y=Y, Z=Z, rstride=1, cstride=1,
                cmap=self.colormap, linewidth=linewidth, antialiased=True, alpha=alpha,
                *args, **kwargs)

    def wireframe(self, x, y, z=None, *args, **kwargs):
        X, Y, Z = self._3d_preprocess(x, y, z)
        self.canvas.plot_wireframe(X=X, Y=Y, Z=Z, color=next(self.colors), *args, **kwargs)

    def projection(self, x, y, z=None, alpha=0.1, linewidth=0, *args, **kwargs):
        X, Y, Z = self._3d_preprocess(x, y, z)
        xmin = np.min(x)
        ymin = np.min(Y)
        zmin = np.min(Z)
        self.canvas.plot_surface(X, Y, Z, rstride=1, cstride=1,
                linewidth=linewidth, alpha=alpha, cmap=self.colormap,
                *args, **kwargs)
        self.canvas.contour(X, Y, Z, zdir='z', offset=zmin, cmap=self.colormap)
        self.canvas.contour(X, Y, Z, zdir='x', offset=xmin, cmap=self.colormap)
        self.canvas.contour(X, Y, Z, zdir='y', offset=ymin, cmap=self.colormap)

    def set_camera(self, elev=None, azim=None):
        self.canvas.view_init(elev, azim)

    def set_axis(self, xtitle='', ytitle='', ztitle=''):
        self.canvas.set_xlabel(xtitle, labelpad=12)
        self.canvas.set_ylabel(ytitle, labelpad=12)
        self.canvas.set_zlabel(ztitle, labelpad=12)


class Container(Plot):

    def __init__(self, rows=1, cols=2, height=None, width=None, *args, **kwargs):
        if height is None:
            height = 2600.0 * rows
        if width is None:
            width = 4200.0 * cols
        super(Container, self).__init__(height=height, width=width, *args, **kwargs)
        self.canvas.axis('off')
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)
        self.canvases = []
        self.plots = []
        for i in range(rows):
            self.plots.append([])
            self.canvases.append([])
            for j in range(cols):
                subplot = self.figure.add_subplot(rows, cols, i * cols + j + 1)
                subplot.axis('off')
                self.plots[i].append(None)
                self.canvases[i].append(subplot)
#        self.figure.set_tight_layout(False)

    def _preprint(self):
        for i in range(self.rows):
            for j in range(self.cols):
                plot = self.plots[i][j]
                canvas = self.canvases[i][j]
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
