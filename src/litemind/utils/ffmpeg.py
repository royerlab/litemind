"""Utilities for configuring ffmpeg binary paths from imageio-ffmpeg."""


def _set_ffmpeg_binary():
    """
    Configure ffmpeg and ffprobe binary paths from imageio-ffmpeg.

    Sets the ``FFMPEG_BINARY`` and ``FFPROBE_BINARY`` environment variables,
    creates symlinks in ``~/local/bin/``, and adds that directory to PATH.

    Raises
    ------
    FileNotFoundError
        If the ffmpeg executable provided by imageio-ffmpeg is not found.
    """
    import os

    import imageio_ffmpeg

    # Get the path of the ffmpeg executable provided by imageio-ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    # Check if the path is valid
    if not os.path.isfile(ffmpeg_path):
        raise FileNotFoundError(f"ffmpeg executable not found at {ffmpeg_path}")

    # Set the FFMPEG_BINARY environment variable
    os.environ["FFMPEG_BINARY"] = ffmpeg_path

    # Print the path to verify
    print(f"FFMPEG_BINARY set to: {os.environ['FFMPEG_BINARY']}")

    # Set the FFPROBE_BINARY environment variable to the same path as ffmpeg
    os.environ["FFPROBE_BINARY"] = ffmpeg_path

    # Print the path to verify
    print(f"FFPROBE_BINARY set to: {os.environ['FFPROBE_BINARY']}")

    # Create a local bin directory if it doesn't exist
    local_bin_dir = os.path.expanduser("~/local/bin")
    os.makedirs(local_bin_dir, exist_ok=True)

    # Create a symlink to ffmpeg in the local bin directory
    symlink_path = os.path.join(local_bin_dir, "ffmpeg")
    if not os.path.exists(symlink_path):
        os.symlink(ffmpeg_path, symlink_path)
        print(f"Symlink created at: {symlink_path}")

    # Create a symlink to ffprobe in the local bin directory
    symlink_path_ffprobe = os.path.join(local_bin_dir, "ffprobe")
    if not os.path.exists(symlink_path_ffprobe):
        os.symlink(ffmpeg_path, symlink_path_ffprobe)
        print(f"Symlink created at: {symlink_path_ffprobe}")

    # Add the local bin directory to the PATH environment variable
    os.environ["PATH"] = local_bin_dir + os.pathsep + os.environ["PATH"]
    print(f"PATH updated to include: {local_bin_dir}")
