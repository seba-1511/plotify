
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
    font_manager.fontManager.addfont(path)


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
    for font_file in font_manager.findSystemFonts(
        fontpaths=[dir_path],
        fontext="ttf",
    ):
        font_manager.fontManager.addfont(font_file)


def install_google_font():
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
    pass


add_ttf_directory(ttf_opensans.opensans().path.parent.absolute())
