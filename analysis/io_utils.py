"""Small input/output checks shared by the standalone figure commands."""
from pathlib import Path


def prepare_output(output, inputs):
    """Keep generated files outside every supplied input directory."""
    output = Path(output).resolve()
    for item in inputs:
        item = Path(item).resolve(strict=True)
        directory = item if item.is_dir() else item.parent
        if output == directory or directory in output.parents:
            raise ValueError("--output-dir must be outside each input directory")
    output.mkdir(parents=True, exist_ok=True)
    return output


def require_columns(frame, columns, filename):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{filename}: missing columns {sorted(missing)}")
