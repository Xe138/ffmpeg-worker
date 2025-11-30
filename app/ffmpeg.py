import shlex
from pathlib import Path

# FFmpeg options that take a value (not exhaustive, covers common ones)
OPTIONS_WITH_VALUES = {
    "-c", "-c:v", "-c:a", "-b", "-b:v", "-b:a", "-r", "-s", "-ar", "-ac",
    "-f", "-t", "-ss", "-to", "-vf", "-af", "-filter:v", "-filter:a",
    "-preset", "-crf", "-qp", "-profile", "-level", "-pix_fmt", "-map",
    "-metadata", "-disposition", "-threads", "-filter_complex",
}


def parse_command(command: str) -> list[str]:
    """Parse FFmpeg command string into argument list."""
    return shlex.split(command)


def resolve_paths(args: list[str], data_path: str) -> list[str]:
    """Resolve relative paths against the data directory."""
    resolved = []
    skip_next = False

    for i, arg in enumerate(args):
        if skip_next:
            resolved.append(arg)
            skip_next = False
            continue

        # Check if this is an option that takes a value
        if arg in OPTIONS_WITH_VALUES or arg.startswith("-"):
            resolved.append(arg)
            if arg in OPTIONS_WITH_VALUES:
                skip_next = True
            continue

        # This looks like a file path - resolve if relative
        path = Path(arg)
        if not path.is_absolute():
            resolved.append(str(Path(data_path) / arg))
        else:
            resolved.append(arg)

    return resolved
