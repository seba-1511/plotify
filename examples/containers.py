#!/usr/bin/env python3

from plotify import Plot, Container


def set_all(cont, plot):
    for i in range(cont.rows):
        for j in range(cont.cols):
            cont.set_plot(i, j, plot)


if __name__ == '__main__':
    native = Plot('Native')
    native.set_axis('x-axis', 'y-axis')
    square = Plot(title='Square', width=3000, height=3000)
    square.set_axis('x-axis', 'y-axis')
    for i in range(1, 6, 2):
        for j in range(1, 6, 2):
            cont = Container(i, j, title='Square')
            set_all(cont, square)
            name = './outputs/containers/square' + str(i) + 'x' + str(j) + '.pdf'
            cont.save(name)
            print(name)
            cont.close()

            cont = Container(i, j, title='Native')
            set_all(cont, native)
            name = './outputs/containers/native' + str(i) + 'x' + str(j) + '.pdf'
            cont.save(name)
            print(name)
            cont.close()
