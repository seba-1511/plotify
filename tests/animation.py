#!/usr/bin/env python

import numpy as np
from plotify import Plot, Drawing, Plot3D, Animation

if __name__ == '__main__':
    x = np.linspace(-7, 7, 50)
    fn = lambda x, y: - np.sin(x / 2.0) + y**2

    # Setup 3D Plot
    p = Plot3D('3D Rotation')
    p.projection(x, np.cos(x + 0.5), fn)
    p.set_axis('x axis', 'y axis', 'z axis')
    p.set_camera(45, 66)

    # 3D Rotation
    a3d = Animation()
    a3d.rotate_3d(p)
    a3d.save('./outputs/rot3d.gif')
    # a3d.save('./rot3d.mp4')

    # Setup Scatter Graph
    x = np.linspace(-7, 7, 30)
    q = Plot('Scatter (PuBuGn_d)', 500, 500, 100)
    q.canvas.axis('off')

    # Scatter animation
    def fn(d):
        q.scatter(d, np.cos(0.7 + d))
        return q
    a = Animation()
    a.anim_fn(fn, x)
    a.save('./output/scatter.gif')
    a.save('./outputs/scatter.mp4')

