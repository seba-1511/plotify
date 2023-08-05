
import colorsys
import matplotlib.colors as mc


LIGHT_GRAY = '#D3D3D3'
MAUREENSTONE_COLORS = [
    '#396AB1',
    '#DA7C30',
    '#3E9651',
    '#CF2529',
    '#535154',
    '#6B4C9A',
    '#922428',
    '#F8B620',
    '#E377C2',
]
Maureen = {
    'blue': MAUREENSTONE_COLORS[0],
    'orange': MAUREENSTONE_COLORS[1],
    'green': MAUREENSTONE_COLORS[2],
    'red': MAUREENSTONE_COLORS[3],
    'black': MAUREENSTONE_COLORS[4],
    'purple': MAUREENSTONE_COLORS[5],
    'cardinal': MAUREENSTONE_COLORS[6],
    'gold': MAUREENSTONE_COLORS[7],
    'pink': MAUREENSTONE_COLORS[8],
}
Vibrant = {
    'cyan': '#33BBEE',
    'magenta': '#EE3377',
    'teal': '#009988',
    'orange': '#EE7733',
    'blue': '#0077BB',
    'red': '#CC3311',
    'gray': '#BBBBBB',
}
Google = {
    'red': '#EA4335',
    'yellow': '#FBBC04',
    'green': '#34A853',
    'blue': '#4285F4',
    'gray': '#9AA0A6',
    'orange': '#FA7B17',
    'pink': '#F439A0',
    'purple': '#A142F4',
    'cyan': '#24C1E0',
}


def lighten_color(color, amount=0.5):
    """
    ## Description

    Lightens a color by a given amount.

    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    The higher the amount, the lighter the color.

    ## References

    Taken from:

    [StackOverflow](https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib/49601444)

    ## Example

    ~~~python
    lighten_color('g', 0.3)
    lighten_color('#F034A3', 0.6)
    lighten_color((.3,.55,.1), 0.5)
    ~~~

    """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
