#!/usr/bin/env python3

from plotify import Plot, Container


class PublicationPlot(Plot):

    def __init__(self, title='', height=2600.0, width=2600.0, dpi=530.0, *args, **kwargs):
        super(PublicationPlot, self).__init__(title, height, width, dpi, *args, **kwargs)
        self.set_title(title)
        self.stretch(top=-0.02, left=0.15, right=0.05, bottom=0.1)
        self._outset_bbox_to_anchor = {
            'upper center': (0.5, 1.32),
            'lower center': (0.5, -0.45),
            'upper right': (1.7, 1.035),
            'lower right': (1.7, -0.035),
        }

    def set_title(self, title, loc='left', x=0.27, y=0.97, text_obj=None):
        super(PublicationPlot, self).set_title(title, loc, x, y, text_obj)

    def set_subtitle(self, title, loc='right', x=0.95, y=0.92):
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

    def set_title(self, title, loc='left', x=0.12, y=0.98, text_obj=None):
        super(LowResPlot, self).set_title(title, loc=loc, x=x, y=y, text_obj=text_obj)

    def set_subtitle(self, title, loc='right', x=0.9, y=0.92):
        super(LowResPlot, self).set_subtitle(title, loc=loc, x=x, y=y)

    def save(self, *args, bbox_inches='tight', **kwargs):
        super(LowResPlot, self).save(*args, bbox_inches=bbox_inches, **kwargs)


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
