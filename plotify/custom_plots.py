#!/usr/bin/env python3

from plotify import Plot, Container


class PublicationPlot(Plot):

    def __init__(self, title='', height=2600.0, width=2600.0, dpi=530.0, *args, **kwargs):
        super(PublicationPlot, self).__init__(title, height, width, dpi, *args, **kwargs)
        self.set_title(title)
        self.stretch(top=-0.02, left=0.15, right=0.05, bottom=0.1)

    def set_title(self, title, loc='left', x=0.26, y=0.97, text_obj=None):
        super(PublicationPlot, self).set_title(title, loc, x, y, text_obj)

    def set_subtitle(self, title, loc='right', x=0.96, y=0.92):
        super(PublicationPlot, self).set_subtitle(title, loc=loc, x=x, y=y)


class LowResPlot(Plot):

    def __init__(self, title='', height=975.0, width=1800.0, dpi=210.0, *args, **kwargs):
        super(LowResPlot, self).__init__(title,
                                         height,
                                         width,
                                         dpi,
                                         *args,
                                         **kwargs)

    def save(self, *args, bbox_inches='tight', **kwargs):
        super(LowResPlot, self).save(*args, bbox_inches=bbox_inches, **kwargs)


class ListContainer(Container):

    def __init__(self, list_plots, *args, **kwargs):
        rows = len(list_plots)
        cols = len(list_plots[0])
        height = list_plots[0].height
        width = list_plots[0].width
        super(ListContainer, self).__init__(rows=rows,
                                            cols=cols,
                                            height=height,
                                            width=width,
                                            *args,
                                            **kwargs)

        for i, row in enumerate(list_plots):
            for j, plot in enumerate(row):
                self.set_plot(i, j, plot)
