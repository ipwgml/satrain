"""
satrain.config
===============

Provides functionality to manage the local configuration of the satrain package.

The local configuration of the ``satrain`` package contains only a single entry
named 'data_path' and specifies the location at which the training and testing
data is stored.

In order to conserve the ``satrain`` configuration between sessions, ``satrain`` will
create a ``config.toml`` in the user's configuration folder. By default, the 'data_path'
will be read from this file. However, the 'data_path' read from the configuration file
will be overwritten by the ``SATRAIN_DATA_PATH`` environment variable.
"""
import logging
from pathlib import Path
import os

import appdirs
import rich
import toml


LOGGER = logging.getLogger(__name__)


CONFIG_DIR = Path(appdirs.user_config_dir("satrain", "ipwg"))


def get_data_path() -> Path:
    """
    Get the root of the SatRain data path.

    The satrain data path is determined as follows:
        1. In the absence of any configuration, it defaults to the current working directory.
        2. If a satrain config file exists and it contains a 'data_path' entry, this will replace
           the current working directory determined in the previous step.
        3. Finally, if the 'SATRAIN_DATA_PATH' environment variable is set, it will overwrite the
           settings from the config file.

    Return:
        A Path object pointing to the root of the satrain data path.
    """
    # Default value is current working directory.
    data_path = Path(os.getcwd())

    # If config file exists, try to parse 'data_path' from it.
    config_file = CONFIG_DIR / "config.toml"
    if config_file.exists():
        try:
            config = toml.loads(open(config_file, "r").read())
        except Exception:
            LOGGER.exception(
                "Encountered an error when trying the read the config file located as %s",
                config_file
            )
        new_data_path = config.get("data_path", None)
        if new_data_path is None:
            LOGGER.warning(
                "satrain config file exists at %s but it doesn't contain a 'data_path' entry.",
                new_data_path
            )
        else:
            data_path = Path(new_data_path)
    else:
        if "SATRAIN_DATA_PATH" not in os.environ:
            LOGGER.warning(
                "Initializing the SATRAIN_DATA_PATH to %s. Use 'satrain config set_data_path' "
                " to customize the location of the SatRain data.",
                data_path
            )
            set_data_path(data_path)

    # Finally, check if environment variable is set.
    new_data_path = os.environ.get("SATRAIN_DATA_PATH", None)
    if new_data_path is not None:
        if Path(new_data_path) != data_path:
            LOGGER.warning(
                "Environment variable SATRAIN_DATA_PATH"
            )

        data_path = Path(new_data_path)

    return data_path


def set_data_path(path: str | Path) -> None:
    """
    Set data path and write data path to 'satrain' config file.

    Args:
        path: A string or Path or path object specifying the satrain data path.
    """
    CONFIG_DIR.mkdir(exist_ok=True, parents=True)
    config_file = CONFIG_DIR / "config.toml"
    config = {"data_path": str(path)}
    with open(config_file, "w") as output:
        output.write(toml.dumps(config))

def show() -> None:
    """
    Display configuration information.
    """
    current_data_path = str(get_data_path())

    config_file = CONFIG_DIR / "config.toml"
    if config_file.exists():
        config_file = str(config_file)
    else:
        config_file = "None"

    satrain_data_path = os.environ.get("SATRAIN_DATA_PATH")
    if satrain_data_path is None:
        satrain_data_path = "NOT SET"

    rich.print(
        f"""
[bold red]satrain config [/bold red]

Current data path: [bold red]{current_data_path}[/bold red]
Config file: {config_file}
Environment variable SATRAIN_DATA_PATH: {satrain_data_path}
        """
    )
