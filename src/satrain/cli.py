"""
satrain.cli
===========

Provides a command-line interface (CLI) for managing the satrain configuration and
downloading data.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List

import click
import rich
from rich.table import Table

from satrain.config import (
    show,
    set_data_path,
    get_data_path
)
from satrain import data
import satrain.logging


LOGGER = logging.getLogger(__name__)


@click.group
def satrain():
    """Command line interface for the 'satrain' package."""


#
# satrain config
#

@satrain.group()
def config():
    """
    Configure the satrain package for the current user.
    """

config.command(show)

@config.command(name="set_data_path")
@click.argument("path")
def set_data_path(path: str):
    """Set the satrain data path."""
    from satrain.config import set_data_path
    set_data_path(path)

#
# satrain download
#


satrain.add_command(data.cli, name="download")


def flatten(dict_or_list: List[Path] | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(dict_or_list, list):
        return len(dict_or_list)
    if isinstance(dict_or_list, dict):
        flattened = {}
        for name, value in dict_or_list.items():
            value_flat = flatten(value)
            if isinstance(value_flat, int):
                flattened[name] = value_flat
            else:
                for name_flat, files in value_flat.items():
                    flattened[name + "/" + name_flat] = files
        return flattened


@satrain.command(name="list_files")
def list_files():
    """
    List locally available satrain file.
    """
    current_data_path = str(get_data_path())

    rich.print(f"""
Data path: {current_data_path}
    """)

    table = Table(title="satrain files")
    table.add_column("Relative path", justify="left")
    table.add_column("# files", justify="right")

    file_cts = {}
    files = flatten(data.list_local_files())
    for rel_path, n_files in files.items():
        key = Path(rel_path).parent.parent.parent
        file_cts[key] = file_cts.setdefault(key, 0) + n_files

    for rel_path, n_files in file_cts.items():
        table.add_row("satrain/" + str(rel_path), str(n_files))

    rich.print(table)
