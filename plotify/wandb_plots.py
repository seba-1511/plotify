#!/usr/bin/env python

import plotify as pl
import cherry as ch
import wandb


def wandb_plot(config):
    """
    Generates a new plot from a configuration dictionary.

    Example configuration:

    config = {
        'type': pl.PublicationPlot,
        'title': 'PT Tasks',
        'subtitle': 'mini-ImageNet',
        'xtitle': 'Iterations',
        'ytitle': 'Accuracy',
        'xlims': (0, 1000),
        'ylims': (0.0, 1.0),
        'legend': {
            'inset': True,
            'loc': 'best',
            'show': True,
        },
        'results': [  # Each dict is a curve
            {
                'wandb_id': 'arnolds/meta-features/3dhgg3yw',
                'x_key': 'iteration',
                'y_key': 'valid/accuracy',
                'label': 'Varnish',
                'color': pl.Maureen['blue'],
                'temperature': 50.0,
                'markevery': 100,
            },
            {
                'wandb_id': 'arnolds/meta-features/11r23eby',
                'x_key': 'iteration',
                'y_key': 'valid/accuracy',
                'label': 'No Varnish',
                'color': pl.Maureen['orange'],
                'temperature': 50.0,
                'markevery': 100,
            },
        ],
    }
    """

    # Enable LaTeX
    pl.usetex()

    # Setup plot
    plot = config.get('type', pl.LowResPlot)()
    plot.set_title(config.get('title'))
    plot.set_subtitle(config.get('subtitle'))
    plot.set_axis(config.get('xtitle'), config.get('ytitle'))
    plot.set_lims(config.get('xlims'), config.get('ylims'))
    plot.set_notation(config.get('notation', 'scientific'))
    plot.set_legend(**config.get('legend', {}))

    # Plot the results
    api = wandb.Api()
    for result in config.get('results'):
        run = api.run(result.get('wandb_id'))
        x_key = result.get('x_key', '_step')
        y_key = result.get('y_key')
        values = run.history(keys=[x_key, y_key], samples=result.get('samples', 2048))
        x_values = values[x_key].to_numpy()
        y_values = values[y_key].to_numpy()
        # TODO: Cut y_values to y_lims here (before smoothing)
        # TODO: handle list of runs
        x_values, y_values = ch.plot.smooth(
            x=x_values,
            y=y_values,
            temperature=result.get('temperature', 1.0)
        )
        plot.plot(
            x=x_values,
            y=y_values,
            label=result.get('label'),
            color=result.get('color'),
            linestyle=result.get('linestyle'),
            linewidth=result.get('linewidth'),
            markevery=result.get('markevery'),
        )
        # TODO: for list of runs, compute ci95.

    return plot
