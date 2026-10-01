"""Reading and writing of the JSON files used by the program."""

import json
import os
from typing import Any, Dict, List

from src.models import FunctionDefinition, PromptInput


def read_json_list(path: str) -> List[Any]:
    """Read a JSON file that must contain an array.

    Args:
        path: Path to the file.

    Returns:
        The decoded array.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the content is not valid JSON or not an array.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("expected a JSON array at the top level")
    return data


def load_functions(path: str) -> List[FunctionDefinition]:
    """Load and validate the function definitions.

    Args:
        path: Path to the functions definition file.

    Returns:
        The validated function definitions.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the content is invalid or has duplicate names.
    """
    functions = [
        FunctionDefinition.model_validate(item)
        for item in read_json_list(path)
    ]
    if not functions:
        raise ValueError("no function is defined")
    names = [fn.name for fn in functions]
    if len(set(names)) != len(names):
        raise ValueError("function names must be unique")
    return functions


def load_prompts(path: str) -> List[PromptInput]:
    """Load and validate the test prompts.

    Args:
        path: Path to the prompts file.

    Returns:
        The validated prompts.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the content is invalid.
    """
    return [
        PromptInput.model_validate(item) for item in read_json_list(path)
    ]


def write_results(path: str, results: List[Dict[str, Any]]) -> None:
    """Write the results as JSON, creating the parent directory if needed.

    Args:
        path: Output file path.
        results: Function calls to write.

    Raises:
        OSError: If the file cannot be written.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")


def describe_error(path: str, exc: Exception) -> str:
    """Build a clear message for an error raised while reading a file.

    Args:
        path: The file being read.
        exc: The exception that was raised.

    Returns:
        A one-line message for the user.
    """
    if isinstance(exc, FileNotFoundError):
        return f"file not found: {path}"
    if isinstance(exc, json.JSONDecodeError):
        return f"invalid JSON in {path}: {exc}"
    if isinstance(exc, OSError):
        return f"cannot read {path}: {exc}"
    return f"invalid content in {path}: {exc}"
