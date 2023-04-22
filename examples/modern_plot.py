
import plotify as pl
import numpy as np

plot = pl.ModernPlot()
plot.set_palette('vibrant')
plot.set_axis('Iterations', r'Values $(\times 1,000)$')
plot.set_grid('horizontal', granularity='fine')
plot.set_notation(x='scientific', y='decimal')

# title
plot.set_title('mini-ImageNet')

# subtitle
plot.set_subtitle(r'$\int f(x) dx$')

# plot data
num_points = 10
plot.plot(
    np.arange(num_points),
    np.arange(num_points)**1 / 1000,
    label=r'Linear',
)
plot.plot(
    np.arange(num_points),
    np.arange(num_points)**2 / 1000,
    label='Quadratic',
)
plot.plot(
    np.arange(num_points),
    np.arange(num_points)**3 / 1000,
    label='Cubic',
)
plot.plot(
    np.arange(num_points),
    np.exp(np.arange(num_points)) / 1000,
    label='Exponential',
)

plot.set_legend(title='Curve Type')

plot.save('outputs/test.pdf', bbox_inches='tight')
