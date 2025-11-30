import asyncio
import shlex
from pathlib import Path

from app.models import Progress

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


def parse_progress(output: str, duration_seconds: float | None) -> Progress:
    """Parse FFmpeg progress output into Progress model."""
    data: dict[str, str] = {}
    for line in output.strip().split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()

    frame = int(data.get("frame", 0))
    fps = float(data.get("fps", 0.0))
    out_time_ms = int(data.get("out_time_ms", 0))
    bitrate = data.get("bitrate", "0kbits/s")

    # Convert out_time_ms to HH:MM:SS.mm format
    total_seconds = out_time_ms / 1_000_000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

    # Calculate percent if duration is known
    percent = None
    if duration_seconds and duration_seconds > 0:
        percent = (total_seconds / duration_seconds) * 100
        percent = min(percent, 100.0)  # Cap at 100%

    return Progress(
        frame=frame,
        fps=fps,
        time=time_str,
        bitrate=bitrate,
        percent=percent,
    )


def extract_output_path(args: list[str]) -> str | None:
    """Extract output file path from FFmpeg arguments (last non-option argument)."""
    # Work backwards to find the last argument that isn't an option or option value
    i = len(args) - 1
    while i >= 0:
        arg = args[i]
        # Skip if it's an option
        if arg.startswith("-"):
            i -= 1
            continue
        # Check if previous arg is an option that takes a value
        if i > 0 and args[i - 1] in OPTIONS_WITH_VALUES:
            i -= 1
            continue
        # Check if it looks like a file path (has extension or contains /)
        if "." in arg or "/" in arg:
            return arg
        i -= 1
    return None


async def get_duration(input_path: str) -> float | None:
    """Get duration of media file using ffprobe."""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()

        if process.returncode == 0:
            return float(stdout.decode().strip())
    except (ValueError, OSError):
        pass
    return None
