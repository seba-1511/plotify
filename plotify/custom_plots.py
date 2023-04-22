#!/usr/bin/env python3

from matplotlib import font_manager
import matplotlib.patheffects as path_effects

from plotify import Plot, Container


class PublicationPlot(Plot):

    def __init__(self, title='', height=2600.0, width=2600.0, dpi=530.0, *args, **kwargs):
        super(PublicationPlot, self).__init__(
            title, height, width, dpi, *args, **kwargs)
        self.set_title(title)
        self.stretch(top=-0.02, left=0.15, right=0.05, bottom=0.1)
        self._outset_bbox_to_anchor = {
            'upper center': (0.5, 1.32),
            'lower center': (0.5, -0.45),
            'upper right': (1.7, 1.035),
            'lower right': (1.7, -0.035),
        }

    def set_title(self, title, loc='left', x=0.27, y=0.95, **kwargs):
        super(PublicationPlot, self).set_title(
            title=title, loc=loc, x=x, y=y, **kwargs)

    def set_subtitle(self, title, loc='right', x=0.95, y=0.91):
        super(PublicationPlot, self).set_subtitle(title, loc=loc, x=x, y=y)


class LowResPlot(Plot):

    def __init__(self, title='', height=975.0, width=1800.0, dpi=210.0, *args, **kwargs):
        super(LowResPlot, self).__init__(title,
                                         height,
                                         width,
                                         dpi,
                                         *args,
                                         **kwargs)
        self._outset_bbox_to_anchor = {
            'upper center': (0.5, 1.17),
            'lower center': (0.5, -0.40),
            'upper right': (1.35, 1.035),
            'lower right': (1.35, -0.035),
        }

    def set_title(self, title, loc='left', x=0.12, y=0.98, **kwargs):
        super(LowResPlot, self).set_title(title, loc=loc, x=x, y=y, **kwargs)

    def set_subtitle(self, title, loc='right', x=0.9, y=0.92):
        super(LowResPlot, self).set_subtitle(title, loc=loc, x=x, y=y)

    def save(self, *args, bbox_inches='tight', **kwargs):
        super(LowResPlot, self).save(*args, bbox_inches=bbox_inches, **kwargs)


class ModernPlot(PublicationPlot):

    def __init__(self, *args, **kwargs):
        super(ModernPlot, self).__init__(*args, **kwargs)
        self.set_font('Open Sans')
        self.update_settings({
            'font.variant': 'normal',
            'font.weight': 'light',
            'legend.fancybox': False,
            'legend.title_fontsize': 13,
            'legend.fontsize': 14,
            'axes.labelsize': 13,
            'axes.labelweight': 'semibold',
            'xtick.labelsize': 14,
            'xtick.minor.visible': False,
            'ytick.labelsize': 14,
            'ytick.minor.visible': False,
        })
        axis_label_font = font_manager.FontProperties(
            family='Open Sans',
            size=13,
            weight='semibold',
        )
        self.canvas.xaxis.label.set_fontproperties(axis_label_font)
        self.canvas.yaxis.label.set_fontproperties(axis_label_font)

    def _preprint(self):
        self._legend_options['title_fontproperties'] = {
            'weight': 'bold',
        }
        ticks_font = font_manager.FontProperties(
            family=self.settings['font.sans-serif'],
            size=17,
            weight='light',
        )
        for label in self.canvas.get_xticklabels():
            label.set_fontproperties(ticks_font)
        for label in self.canvas.get_yticklabels():
            label.set_fontproperties(ticks_font)
        return super(ModernPlot, self)._preprint()

    def set_title(self, title, x=0.273, y=0.885, **kwargs):
        super(ModernPlot, self).set_title(
            title=title,
            x=x,
            y=y,
            **kwargs,
        )
        text_obj = kwargs.get('text_obj', self.title)
        text_obj.set_fontproperties(
            font_manager.FontProperties(
                size=20,
                weight='light',
            )
        )
        text_obj.set_va('baseline')
        text_obj.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='white'),
            path_effects.Normal(),
        ])

    def set_subtitle(self, title, x=0.95, y=0.885, loc='right', *args, **kwargs):
        self.set_title(
            title=title,
            text_obj=self.subtitle,
            x=x,
            y=y,
            loc='right',
            *args,
            **kwargs,
        )
        self.subtitle.set_fontproperties(
            font_manager.FontProperties(
                size=16,
                weight='light',
            )
        )


class ListContainer(Container):

    def __init__(self, list_plots, *args, **kwargs):
        rows = len(list_plots)
        cols = len(list_plots[0])
        height = kwargs.pop('height', list_plots[0][0].height)
        width = kwargs.pop('width', list_plots[0][0].width)
        dpi = kwargs.pop('dpi', list_plots[0][0].dpi)
        super(ListContainer, self).__init__(rows=rows,
                                            cols=cols,
                                            height=height,
                                            width=width,
                                            dpi=dpi,
                                            *args,
                                            **kwargs)

        for i, row in enumerate(list_plots):
            for j, plot in enumerate(row):
                self.set_plot(i, j, plot)
