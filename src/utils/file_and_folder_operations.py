import json
import os
import pickle  # nosec B403
from pathlib import Path
from typing import Any, Optional


def load_pickle(file: str, mode: str = "rb") -> Any:
    """Load a pickled object from a file.

    Args:
        file: The path to the file containing the pickled object.
        mode: The mode in which the file is opened. Defaults to 'rb'.

    Returns:
        The unpickled object.
    """
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def save_pickle(obj: Any, file: str, mode: str = "wb") -> None:
    """Save an object to a file using pickle.

    Args:
        obj: The object to be pickled and saved.
        file: The path to the file where the object will be saved.
        mode: The mode in which the file is opened. Defaults to 'wb'.
    """
    with open(file, mode) as f:
        pickle.dump(obj, f)


def load_json(file: str) -> Any:
    """Load a JSON object from a file.

    Args:
        file: The path to the file containing the JSON object.

    Returns:
        Any: The loaded JSON object.
    """
    with open(file) as f:
        a = json.load(f)
    return a


def save_json(obj: Any, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    """Save an object to a file in JSON format.

    Args:
        obj: The object to be saved in JSON format.
        file: The path to the file where the JSON will be saved.
        indent: The number of spaces to use for indentation. Defaults to 4.
        sort_keys: Whether to sort the keys in the output. Defaults to True.
    """
    with open(file, "w") as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def subdirs(
    folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True
) -> list[str]:
    """Get a list of subdirectories in a folder.

    Args:
        folder: The path to the folder.
        join: Whether to join the folder path with subdirectory names. Defaults to True.
        prefix: Filter subdirectories by prefix. Defaults to None.
        suffix: Filter subdirectories by suffix. Defaults to None.
        sort: Whether to sort the resulting list. Defaults to True.

    Returns:
        A list of subdirectory names in the given folder.
    """
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E731, E741
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def subfiles(
    folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True
) -> list[str]:
    """Get a list of files in a folder.

    Args:
        folder: The path to the folder.
        join: Whether to join the folder path with file names. Defaults to True.
        prefix: Filter files by prefix. Defaults to None.
        suffix: Filter files by suffix. Defaults to None.
        sort: Whether to sort the resulting list. Defaults to True.

    Returns:
        A list of file names in the given folder.
    """
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E731, E741
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def remove_suffixes(filename: Path) -> Path:
    """Remove all the suffixes from a filename.

    Args:
        filename (Path): The Path object representing the filename.

    Returns:
        Path: The filename without its extensions.
    """
    return Path(str(filename).removesuffix("".join(filename.suffixes)))
