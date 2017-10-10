#!/usr/bin/env python3

import numpy as np
from plotify import Plot, Drawing, Plot3D, Container, Image

if __name__ == '__main__':
    x = np.linspace(-7, 7, 50)
    fn = lambda x, y: -np.sin(x/2.0) + y**2

    p = Plot3D('Title')
    p.set_title('3D Plotting (default)')
    p.set_title('$ f(x, y) = y^2 - \sin(\\' + 'frac{1}{2}\cdot x)$')
    p.projection(x, np.cos(x + 0.5), fn)
    # p.surface(x, np.cos(x + 0.5), fn)
    # p.plot(x, np.cos(x), fn, label='cos')
    # p.scatter(x, np.sin(x), fn, label='sin')
    # p.plot(x, np.sin(x + 0.5), label='.5 + sin', jitter=0.1)
    p.set_axis('x axis', 'y axis', 'z axis')
    p.set_camera(45, 66)
    # p.show()
    p.save('./outputs/plot3d.pdf')

    x = np.linspace(-7, 7, 30)
    q = Plot('Scatter (PuBuGn_d)', 500, 500, 100)
    q.set_palette('PuBuGn_d')
    q.scatter(x=x, y=np.cos(x), label='cos')
    q.scatter(x=x, y=np.sin(x), label='sin')
    q.scatter(x, np.sin(x + 0.5), label='.5 + sin')
    # q.show()
    q.save('./outputs/plot.png')

    r = Plot('Bars (husl)')
    r.set_palette('husl')
    # r = Plot('Bar', 500, 500, 150)
    r.bar([1, 2, 3, 4])
    r.bar([[1, 2], [2, 3], [3, 4], [4, 5]])
    # r.show()
    r.save('./outputs/bars.pdf')


    s = Plot('Plotting (custom)')
    s.set_palette('maureen')
    s.plot(x, np.cos(x), label='cos')
    s.plot(x, np.sin(x), label='sin', jitter=0.1)
    s.plot(x, np.sin(x + 0.5), label='.5 + sin', smooth_window=1)
    s.plot(x, np.cos(x + 0.5), label='.5 + cos', jitter=0.1)
    s.set_axis('x-axis', 'y-axis')
    s.save('./outputs/smooth_and_jitter.png')


    t = Plot('Shapes and Text')
    t.set_scales('linear', 'linear')
    t.set_lims(y=(-4.0, 4.0))
    t.plot(x, np.cos(x), label='cos')
    t.circle(0, np.cos(0), radius=0.5)
    t.circle(2, np.cos(2), radius=0.5, color='red', fill=True, linewidth=1.5)
    t.circle(-2, np.cos(-2), radius=0.5, fill='#396AB1')
    t.text('Random text $\cos(x)^2$', (4, -2.7))
    t.annotate('Trust Region', (2.5, 2.7), (2.2, 0.2))
    t.annotate('Trust Region', xytext=(-5.5, -2.2), xylabel=(1.8, -1.2), rad=0.7, shape=']-[')
    t.save('./outputs/shapes.png')


    u = Drawing('Drawing', 3000, 3000)
    u.set_scales('linear', 'linear')
    u.set_lims(x=(-4.0, 4.0), y=(-4.0, 4.0))
    u.plot(x, np.cos(x), label='cos')
    u.rectangle(-2, np.cos(-2), 1.0, 1.0, fill=True)
    u.circle(-2, np.cos(-2), radius=0.5, fill='#396AB1')
    u.rectangle(2, np.cos(2), 1.0, 1.0, fill='#396AB1', alpha=1.0)
    u.circle(2, np.cos(2), radius=0.5, color='red', fill=True, linewidth=1.5)
    u.fancybox(0, np.cos(0), 1.1, 1.1, style='round', fill=True)
    u.circle(0, np.cos(0), radius=0.5)
    u.arrow((3.2, 3.0), (1.2, 1.7), width=1.0, fill='#BF5FFF', alpha=4.0)
    u.save('./outputs/drawing.png')

    i = Image('./outputs/drawing.png')


    cont = Container(3, 3, title='Container')
    cont.set_plot(0, 0, p)
    cont.set_plot(0, 1, q)
    cont.set_plot(0, 2, i)
    cont.set_plot(1, 1, r)
    cont.set_plot(1, 0, s)
    cont.set_plot(2, 0, t)
    cont.set_plot(2, 1, u)
    cont.save('./outputs/container.pdf')
    # cont.show()