
class Marker:

    def __init__(self, symbol='o', fill=True, size=8.5):
        self.symbol = symbol

        if fill:
            self.width = 0.0
            self.color = None
        else:
            self.width = 2.0
            self.color = 'white'

        self.size = size
        if size == 'small':
            self.size = 6.0
            if not fill:
                self.width = 1.0
        elif size == 'large':
            self.size = 8.5
            if fill:
                self.width = 2.0

    def __call__(self, **kwargs):
        new_args = {
            'symbol': kwargs.get('symbol', self.symbol),
            'fill': kwargs.get('fill', self.color is None),
            'size': kwargs.get('size', self.size),
        }
        return Marker(**new_args)


circle = Marker(symbol='o', fill=False, size='large')
cross = Marker(symbol='X', fill=True, size='large')
triangle_down = Marker(symbol='v', fill=False, size='large')
triangle_up = Marker(symbol='^', fill=False, size='large')
triangle_left = Marker(symbol='<', fill=False, size='large')
triangle_right = Marker(symbol='>', fill=False, size='large')
star = Marker(symbol='*', fill=True, size='large')
diamond = Marker(symbol='d', fill=False, size='large')
square = Marker(symbol='s', fill=False, size='large')
pentagone = Marker(symbol='p', fill=False, size='large')

MARKERS = [
    circle,
    cross,
    triangle_down,
    star,
    diamond,
    square,
    triangle_up,
    pentagone,
]
