
class Marker:

    def __init__(self, symbol='o', facecolor=None, edgewidth=2.0):
        self.symbol = symbol
        self.facecolor = facecolor
        self.edgewidth = edgewidth


Circle = Marker(symbol='o', facecolor='white', edgewidth=2.0)
Cross = Marker(symbol='X', facecolor=None, edgewidth=2.0)
DownTriangle = Marker(symbol='v', facecolor='white', edgewidth=2.0)
UpTriangle = Marker(symbol='^', facecolor='white', edgewidth=2.0)
LeftTriangle = Marker(symbol='<', facecolor='white', edgewidth=2.0)
RightTriangle = Marker(symbol='>', facecolor='white', edgewidth=2.0)
Star = Marker(symbol='*', facecolor=None, edgewidth=2.0)
Diamond = Marker(symbol='d', facecolor='white', edgewidth=2.0)
Square = Marker(symbol='s', facecolor='white', edgewidth=2.0)
Pentagone = Marker(symbol='p', facecolor='white', edgewidth=2.0)

MARKERS = [
    Circle,
    Cross,
    DownTriangle,
    Star,
    Diamond,
    Square,
    UpTriangle,
    Pentagone,
]
