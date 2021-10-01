#!/usr/bin/env python

import plotify as pl
import plotify.wandb_plots

config = {
    'type': pl.PublicationPlot,
    'title': 'HalfCheetah-v3',
    'subtitle': 'SAC',
    'xtitle': 'Iterations',
    'ytitle': 'Returns',
    'xlims': (0, 3_000_000),
    'ylims': (0.0, 15000.0),
    'notation': 'scientific',
    'y_notation': 'decimal',
    'legend': {
        'inset': True,
        'loc': 'best',
        'show': True,
    },
    'results': [  # Each dict is a curve
        {
            'wandb_id': [
                'arnolds/qmcrl/51podsf0',
                'arnolds/qmcrl/104w0l8i',
                'arnolds/qmcrl/u3371i4t',
                'arnolds/qmcrl/k86a61gi',
                'arnolds/qmcrl/61inx3b9',
                #  'arnolds/qmcrl/pq7h1956',
                #  'arnolds/qmcrl/5lnjdhyy',
                #  'arnolds/qmcrl/3bktd5yv',
                #  'arnolds/qmcrl/yrpk7li0',
                #  'arnolds/qmcrl/xpouxt8w',
            ],
            'x_key': 'iteration',
            'y_key': 'test/episode_returns',
            'label': 'MC',
            'color': pl.Maureen['blue'],
            'linewidth': 1.8,
            'smooth_temperature': 15.0,
            'markevery': 1000,
            'samples': 4196,
            'shade': 'std',
        },
        {
            'wandb_id': [
                'arnolds/qmcrl/xpouxt8w'
                'arnolds/qmcrl/61inx3b9',
                'arnolds/qmcrl/pq7h1956',
            ],
            'x_key': 'iteration',
            'y_key': 'test/episode_returns',
            'label': 'RQMC',
            'color': pl.Maureen['orange'],
            'linewidth': 1.8,
            'smooth_temperature': 5.0,
            'samples': 512,
            'markevery': 64,
            'shade': 'ci95',
        },
    ],
}

plot = pl.wandb_plots.wandb_plot(config)
plot.save('outputs/wandb_plot.pdf')
