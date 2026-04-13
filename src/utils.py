
import os
import json
import logging
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with a standard format.

    Args:
        name (str): Usually __name__ from the calling file.

    Returns:
        logging.Logger: Ready-to-use logger object.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(name)s  →  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def ensure_dir(path: str) -> None:
    """
    Creates a directory if it doesn't already exist.
    os.makedirs(..., exist_ok=True) does nothing if the folder
    is already there – so it's safe to call multiple times.
    """
    os.makedirs(path, exist_ok=True)


# def save_json(data: list | dict, filepath: str) -> None:
from typing import Union

def save_json(data: Union[list, dict], filepath: str) -> None:
    """
    Saves Python data (list or dict) as a nicely formatted JSON file.
    JSON = JavaScript Object Notation – a universal data format.
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# def load_json(filepath: str) -> list | dict:
from typing import Union

def load_json(filepath: str) -> Union[list, dict]:
    """Loads a JSON file and returns the Python object."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamp_filename(prefix: str, ext: str = "csv") -> str:
    """
    Generates a timestamped filename like:
        articles_2026-04-12_14-30-00.csv
    Useful so old data isn't overwritten every run.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{ts}.{ext}"
