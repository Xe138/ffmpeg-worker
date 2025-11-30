import pytest

from app.ffmpeg import parse_command, resolve_paths


def test_parse_simple_command():
    command = "-i input.mp4 output.mp4"

    args = parse_command(command)

    assert args == ["-i", "input.mp4", "output.mp4"]


def test_parse_command_with_options():
    command = "-i input.mp4 -c:v libx264 -crf 23 output.mp4"

    args = parse_command(command)

    assert args == ["-i", "input.mp4", "-c:v", "libx264", "-crf", "23", "output.mp4"]


def test_parse_command_with_quotes():
    command = '-i "input file.mp4" output.mp4'

    args = parse_command(command)

    assert args == ["-i", "input file.mp4", "output.mp4"]


def test_resolve_paths():
    args = ["-i", "input/video.mp4", "-c:v", "libx264", "output/result.mp4"]
    data_path = "/data"

    resolved = resolve_paths(args, data_path)

    assert resolved == [
        "-i", "/data/input/video.mp4",
        "-c:v", "libx264",
        "/data/output/result.mp4"
    ]


def test_resolve_paths_preserves_absolute():
    args = ["-i", "/already/absolute.mp4", "output.mp4"]
    data_path = "/data"

    resolved = resolve_paths(args, data_path)

    assert resolved == ["-i", "/already/absolute.mp4", "/data/output.mp4"]


def test_resolve_paths_skips_options():
    args = ["-c:v", "libx264", "-preset", "fast"]
    data_path = "/data"

    resolved = resolve_paths(args, data_path)

    # Options and their values should not be resolved as paths
    assert resolved == ["-c:v", "libx264", "-preset", "fast"]
