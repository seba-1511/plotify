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
                'arnolds/qmcrl/pq7h1956',
                'arnolds/qmcrl/5lnjdhyy',
                'arnolds/qmcrl/3bktd5yv',
                'arnolds/qmcrl/yrpk7li0',
                'arnolds/qmcrl/xpouxt8w',
            ],
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

plot = pl.wandb_plots.wandb_plot(config)
plot.save('outputs/wandb_plot.pdf', bbox_inches='tight')
