import pytest

from app.ffmpeg import parse_command, resolve_paths, parse_progress, extract_output_path


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


def test_parse_progress():
    output = """frame=1234
fps=30.24
total_size=5678900
out_time_ms=83450000
bitrate=1250.5kbits/s
progress=continue
"""
    progress = parse_progress(output, duration_seconds=120.0)

    assert progress.frame == 1234
    assert progress.fps == 30.24
    assert progress.time == "00:01:23.45"
    assert progress.bitrate == "1250.5kbits/s"
    assert progress.percent == pytest.approx(69.54, rel=0.01)


def test_parse_progress_no_duration():
    output = "frame=100\nfps=25.0\nout_time_ms=4000000\nbitrate=500kbits/s\n"

    progress = parse_progress(output, duration_seconds=None)

    assert progress.frame == 100
    assert progress.percent is None


def test_extract_output_path():
    args = ["-i", "input.mp4", "-c:v", "libx264", "output.mp4"]

    output_path = extract_output_path(args)

    assert output_path == "output.mp4"


def test_extract_output_path_complex():
    args = ["-i", "a.mp4", "-i", "b.mp4", "-filter_complex", "[0:v][1:v]concat", "out.mp4"]

    output_path = extract_output_path(args)

    assert output_path == "out.mp4"
