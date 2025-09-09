"""
satrain.logging
===============

Configures the logging for the satrain package.
"""
import logging

from rich.logging import RichHandler
from rich.console import Console

_LOG_LEVEL = "INFO"
_CONSOLE = Console()
_HANDLER = RichHandler(
    console=_CONSOLE,
    show_time=False,
    show_path=False
)

# The parent logger for the module.
LOGGER = logging.getLogger("satrain")
logging.basicConfig(
    level=_LOG_LEVEL,
    force=True,
    format="%(message)s",
    handlers=[_HANDLER]
)

def get_console():
    """
    Return the console to use for live logging.
    """
    return _CONSOLE
