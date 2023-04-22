
import os
import ttf_opensans
from matplotlib import font_manager


def add_font(path):
    """
    ## Description

    Adds a font file to Matplotlib.

    ## Arguments

    * `path`: Path to the ttf / otf file.

    ## Example

    ~~~python
    plotify.fonts.add_font('my_file.ttf')
    ~~~
    """
    path = os.path.abspath(os.path.expanduser(path))
    font_manager.fontManager.addfont(path)


def list_fonts():
    font_list = [font.name for font in font_manager.fontManager.ttflist]
    for name in sorted(set(font_list)):
        print(name)


def add_ttf_directory(dir_path):
    """
    ## Description

    Adds all ttf files in a directory

    ## Arguments

    * `dir_path`: Path to the ttf / otf directory.

    ## Example

    ~~~python
    plotify.fonts.add_ttf_directory('my_file.ttf')
    ~~~
    """
    dir_path = os.path.abspath(os.path.expanduser(dir_path))
    for font_file in font_manager.findSystemFonts(
        fontpaths=[dir_path],
        fontext="ttf",
    ):
        add_font(font_file)


add_ttf_directory(ttf_opensans.opensans().path.parent.absolute())
