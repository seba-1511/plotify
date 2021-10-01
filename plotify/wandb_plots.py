#!/usr/bin/env python

import numpy as np
import scipy.interpolate
import plotify as pl
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
    notation = config.get('notation', 'scientific')
    plot.set_notation(
        x=config.get('x_notation', notation),
        y=config.get('y_notation', notation),
    )
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
            run = api.run(run_id)
            values = run.history(keys=[x_key, y_key], samples=samples)
            run_xs = values[x_key].to_numpy()
            run_ys = values[y_key].to_numpy()

            # cut x_values to x_lims here (before smoothing)
            if max_x is not None and max_x < run_xs[-1]:
                cutoff = np.argmax(run_xs > max_x)
                run_xs = run_xs[:cutoff]
                run_ys = run_ys[:cutoff]

            # smooth each run
            if 'smooth_window' in result:
                """
                From:
                https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python/34387987#34387987
                """
                smooth_window = result.get('smooth_window')
                y_cumsum = np.cumsum(run_ys)
                run_ys = (y_cumsum[smooth_window:] - y_cumsum[:-smooth_window]) / smooth_window
                run_xs = run_xs[:-smooth_window]

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

        # interpolate within [runs_min_x, runs_max_x] bounds
        x_linear = np.linspace(
            start=runs_min_x,
            stop=runs_max_x,
            num=samples,
        )
        for run_xs, run_ys in zip(x_values, y_values):
            run_interpolate = scipy.interpolate.interp1d(run_xs, run_ys)
            run_ys[:] = run_interpolate(x_linear)

        # compute mean y
        if len(y_values) > 1:
            y_linear = [np.mean(ys) for ys in zip(*y_values)]
        else:
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

    return plot
