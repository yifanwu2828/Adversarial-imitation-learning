""" colorful console or logger"""

import logging

try:
    import icecream  # noqa
    from icecream import ic

    icecream.install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

try:
    # color console
    from colorama import init, Fore, Back, Style

    init(autoreset=True)
except ImportError:
    from collections import UserString

    class ColoramaMock(UserString):
        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, key):
            return self

    init = ColoramaMock("")
    Back = Cursor = Fore = Style = ColoramaMock("")

# try:
#     # overwrite to more readable error traceback
#     from rich import traceback  # noqa

#     traceback.install()
# except ImportError:
#     print("Fall back to default traceback")


COLORS = {
    # Fore
    "red": Fore.RED,
    "green": Fore.GREEN,
    "blue": Fore.BLUE,
    "yellow": Fore.YELLOW,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE,
    "black": Fore.BLACK,
    "invisible": "",
    # Fore Dim
    "dim_red": Style.DIM + Fore.RED,
    "dim_green": Style.DIM + Fore.GREEN,
    "dim_blue": Style.DIM + Fore.BLUE,
    "dim_yellow": Style.DIM + Fore.YELLOW,
    "dim_magenta": Style.DIM + Fore.MAGENTA,
    "dim_cyan": Style.DIM + Fore.CYAN,
    # Back
    "back_red": Back.RED,
    "back_green": Back.GREEN,
    "back_blue": Back.BLUE,
    "back_yellow": Back.YELLOW,
    "back_magenta": Back.MAGENTA,
    "back_cyan": Back.CYAN,
    # Back Dim
    "back_dim_red": Style.DIM + Back.RED,
    "back_dim_green": Style.DIM + Back.GREEN,
    "back_dim_blue": Style.DIM + Back.BLUE,
    "back_dim_yellow": Style.DIM + Back.YELLOW,
    "back_dim_magenta": Style.DIM + Back.MAGENTA,
    "back_dim_cyan": Style.DIM + Back.CYAN,
    # Back bold
    "back_bold_red": Style.BRIGHT + Back.RED,
    "back_bold_green": Style.BRIGHT + Back.GREEN,
    "back_bold_blue": Style.BRIGHT + Back.BLUE,
    "back_bold_yellow": Style.BRIGHT + Back.YELLOW,
    "back_bold_magenta": Style.BRIGHT + Back.MAGENTA,
    "back_bold_cyan": Style.BRIGHT + Back.CYAN,
}


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors."""

    def __init__(self):
        super().__init__()
        out_fmt = "[%(asctime)s] | %(levelname)s | %(message)s"
        self.log_format = {
            logging.DEBUG: Fore.GREEN + out_fmt,
            logging.INFO: Fore.BLUE + out_fmt,
            logging.WARNING: Fore.YELLOW + out_fmt,
            logging.ERROR: Fore.RED + out_fmt,
            logging.CRITICAL: Back.RED + Fore.WHITE + out_fmt,
        }

    def format(self, record):
        log_fmt = self.log_format.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%H:%M:%S")
        return formatter.format(record)


# Create console (just a logger with color output)
Console = logging.getLogger("")
Console.setLevel(logging.INFO)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())

Console.addHandler(ch)
del ch
