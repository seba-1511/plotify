"""
Main script to create the demo figures in the API docs.
"""

import numpy as np
import plotify as pl


def new_plot():
    return pl.Plot()


if __name__ == "__main__":

    #  # basic
    #  plot = pl.Plot('Title')
    #  plot.save('docs/assets/images/api/basic.png')

    #  # plot
    #  x = np.arange(10)
    #  plot = pl.Plot('Title')
    #  plot.plot(x=x, y=x**2, jitter=5.0, label=r'$f(x) = x^2$', linestyle='dashed')
    #  plot.save('docs/assets/images/api/plot.png')

    #  # scatter
    #  x = np.arange(10)
    #  plot = pl.Plot('Title')
    #  plot.scatter(x=x, y=x**2, label=r'$f(x) = x^2$')
    #  plot.save('docs/assets/images/api/scatter.png')

    #  # errorbar
    #  x = np.arange(10)
    #  plot = pl.Plot('Title')
    #  plot.errorbar(x=x, y=x**2, errors=x, label=r'$f(x) = x^2$')
    #  plot.save('docs/assets/images/api/errorbar.png')

    #  # heatmap
    #  values = np.arange(100).reshape(10, 10)
    #  plot = pl.Plot('Title')
    #  plot.heatmap(
        #  heatvalues=values,
        #  xlabels=[str(x) for x in range(10)],
        #  ylabels=[str(10 - x) for x in range(10)],
        #  show_values=True,
        #  cbar_title='My Color Bar',
    #  )
    #  plot.save('docs/assets/images/api/heatmap.png')

    #  # contour
    #  x = y = np.linspace(-10, 10, num=100)
    #  f = lambda x, y: 0.5 * x**3 + 10.0 * y**2
    #  plot = pl.Plot('Title')
    #  plot.contour(x, y, f, fill=True)
    #  plot.save('docs/assets/images/api/contour.png')

    # set_title
    plot = pl.Plot('Title')
    plot.set_title('New and Much Longer Title', loc='right', x=0.9, y=0.92)
    plot.save('docs/assets/images/api/set_title.png')
