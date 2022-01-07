#!/usr/bin/env python

import numpy as np
import scipy.interpolate
import plotify as pl
import wandb
import math

from .smoothing import smooth


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
                'wandb_id': [
                    'arnolds/qmcrl/51podsf0',
                    'arnolds/qmcrl/xpouxt8w',
                ],
                'x_key': 'iteration',
                'y_key': 'test/episode_returns',
                'label': 'MC',
                'color': pl.Maureen['blue'],
                'linewidth': 1.8,
                'smooth_window': 1,
                'markevery': 1000,
                'samples': 4196,
                'shade': 'std',
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

    # enable LaTeX
    pl.usetex()

    # setup plot
    plot = config.get('type', pl.LowResPlot)()
    plot.set_title(config.get('title'))
    plot.set_subtitle(config.get('subtitle'))
    plot.set_axis(config.get('xtitle'), config.get('ytitle'))
    plot.set_lims(config.get('xlims'), config.get('ylims'))
    plot.set_palette(config.get('palette', 'maureen'))
    x_scale = config.get('x_scale', 'linear')
    y_scale = config.get('y_scale', 'linear')
    plot.set_scales(x=x_scale, y=y_scale)
    notation = config.get('notation', 'scientific')
    if x_scale == 'linear':
        plot.set_notation(x=config.get('x_notation', notation))
    if y_scale == 'linear':
        plot.set_notation(y=config.get('y_notation', notation))
    plot.set_legend(**config.get('legend', {}))

    max_x = config.get('xlims')[1] if 'xlims' in config else None

    # plot the results
    api = wandb.Api()
    for result in config.get('results'):
        x_key = result.get('x_key', '_step')
        y_key = result.get('y_key')
        run_ids = result.get('wandb_id')
        samples = result.get('samples', 2048)

        if not isinstance(run_ids, (list, tuple, set)):
            run_ids = [run_ids, ]

        x_values = []
        y_values = []
        runs_min_x = - float('inf')
        runs_max_x = float('inf')
        for run_id in run_ids:
            try:
                run = api.run(run_id)
                values = run.history(keys=[x_key, y_key], samples=samples)
            except (wandb.errors.CommError, ValueError):
                print('Run not found: ' + run_id + ' - skipping.')
                continue

            try:
                run_xs = values[x_key].to_numpy()
            except KeyError:
                print('Key \'' + x_key + '\' not found for run: ' + run_id + ' - skipping.')
                continue

            try:
                run_ys = values[y_key].to_numpy()
            except KeyError:
                print('Key \'' + y_key + '\' not found for run: ' + run_id + ' - skipping.')
                continue

            # cut x_values to x_lims here (before smoothing)
            if max_x is not None and max_x < run_xs[-1]:
                cutoff = np.argmax(run_xs > max_x)
                run_xs = run_xs[:cutoff]
                run_ys = run_ys[:cutoff]

            # smooth each run
            smooth_temperature = result.get('smooth_temperature', 0.0)
            if smooth_temperature > 0.0:
                run_xs, run_ys = smooth(
                    x=run_xs,
                    y=run_ys,
                    temperature=smooth_temperature,
                )

            # average y values that have the same x values
            xs_increasing = np.diff(run_xs)
            if not np.all(xs_increasing):
                # TODO: implement with scipy lfilter
                new_xs = []
                new_ys = []
                accum = run_ys[0]
                count = 1
                for i in range(1, len(run_xs)):
                    if run_xs[i] > run_xs[i-1]:
                        new_xs.append(run_xs[i-1])
                        new_ys.append(accum / count)
                        accum = run_ys[i]
                        count = 1
                    else:
                        accum += run_ys[i]
                        count += 1
                run_xs = new_xs
                run_ys = new_ys

            # append run result
            x_values.append(run_xs)
            y_values.append(run_ys)
            runs_min_x = max(runs_min_x, np.min(run_xs))
            runs_max_x = min(runs_max_x, np.max(run_xs))

        # plot only if we have values
        if len(x_values) > 0:

            if len(x_values) > 1:
                # interpolate within [runs_min_x, runs_max_x] bounds.
                # needed to ensure all runs are aligned.
                num_interpolate = min(samples, len(run_ys))
                if 'log2' in config.get('x_scale', 'linear'):
                    x_linear = np.logspace(
                        start=math.log(runs_min_x, 2.0),
                        stop=math.log(runs_max_x, 2.0),
                        num=num_interpolate,
                        base=2.0,
                    )
                elif 'log' in config.get('x_scale', 'linear'):
                    x_linear = np.logspace(
                        start=math.log(runs_min_x, 10.0),
                        stop=math.log(runs_max_x, 10.0),
                        num=num_interpolate,
                        base=10.0,
                    )
                else:
                    x_linear = np.linspace(
                        start=runs_min_x,
                        stop=runs_max_x,
                        num=num_interpolate,
                    )
                for run_xs, run_ys in zip(x_values, y_values):
                    run_interpolate = scipy.interpolate.interp1d(run_xs, run_ys)
                    run_ys[:] = run_interpolate(x_linear)

                # compute mean y
                y_linear = [np.mean(ys) for ys in zip(*y_values)]
            else:
                x_linear = x_values[0]
                y_linear = y_values[0]

            # plot curve
            color = result.get('color', next(plot.colors))
            plot.plot(
                x=x_linear,
                y=y_linear,
                label=result.get('label'),
                color=color,
                linestyle=result.get('linestyle'),
                linewidth=result.get('linewidth'),
                markevery=result.get('markevery'),
                marker=result.get('marker'),
                alpha=result.get('alpha'),
            )

            # optionally: show std or ci95
            shade = result.get('shade')
            if shade is not None and len(y_values) > 1:
                y_stds = [np.std(ys) for ys in zip(*y_values)]
                if shade == 'std':
                    y_shade = y_stds
                elif shade == 'ci95':
                    sqrt_n_runs = float(len(y_values))**0.5
                    y_shade = [1.96 * y_std / sqrt_n_runs for y_std in y_stds]
                else:
                    raise ValueError(f'Unknown \'shade\': {shade}')

                y_shade = np.array(y_shade)
                plot.canvas.fill_between(
                    x=x_linear,
                    y1=y_linear - y_shade,
                    y2=y_linear + y_shade,
                    alpha=0.5,
                    color=color,
                    linewidth=0.0,
                )
            # optionally: show error bars
            errorbars = result.get('errorbars')
            if errorbars is not None and len(y_values) > 1:
                y_stds = [np.std(ys) for ys in zip(*y_values)]
                if errorbars == 'std':
                    y_errorbars = y_stds
                elif errorbars == 'ci95':
                    sqrt_n_runs = float(len(y_values))**0.5
                    y_errorbars = [1.96 * y_std / sqrt_n_runs for y_std in y_stds]
                else:
                    raise ValueError(f'Unknown \'errorbars\': {errorbars}')

                y_errorbars = np.array(y_errorbars)
                plot.errorbar(
                    x=x_linear,
                    y=y_linear,
                    errors=y_errorbars,
                    vertical=True,
                    color=color,
                )

    return plot
