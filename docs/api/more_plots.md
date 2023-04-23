
# Custom Plots

The following classes inherit from `BasePlot`.

## Predefined 2D Plots

::: plotify.custom_plots.Plot
    selection:
      members:
        - __init__

::: plotify.custom_plots.PublicationPlot
    selection:
      members:
        - __init__

::: plotify.custom_plots.LowResPlot
    selection:
      members:
        - __init__

::: plotify.custom_plots.ModernPlot
    selection:
      members:
        - __init__

::: plotify.plot.Image
    selection:
      members:
        - __init__

::: plotify.plot.Drawing
    selection:
      members:
        - __init__

## 3D Plots

::: plotify.plot.Plot3D
    selection:
      members:
        - __init__
        - set_camera
        - projection
        - wireframe
        - surface
        - set_notation

## Animations

::: plotify.plot.Animation
    selection:
      members:
        - __init__
        - reset
        - add_frame
        - rotate_3d
        - save

## Containers

::: plotify.plot.Container
    selection:
      members:
        - __init__
        - set_plot
        - get_plot

::: plotify.custom_plots.ListContainer
    selection:
      members:
        - __init__
