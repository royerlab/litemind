import os

from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


def _is_safe_path(base_dir: str, path: str) -> bool:
    """
    Check if extracted path stays within base directory to prevent path traversal attacks.

    Parameters
    ----------
    base_dir : str
        The base directory where extraction should be contained.
    path : str
        The path to validate.

    Returns
    -------
    bool
        True if the path is safe (stays within base_dir), False otherwise.
    """
    abs_base = os.path.abspath(base_dir)
    abs_path = os.path.abspath(os.path.join(base_dir, path))
    return abs_path.startswith(abs_base + os.sep) or abs_path == abs_base


def _safe_extract_zip(zip_ref, dest_dir: str) -> None:
    """Safely extract zip archive with path traversal protection."""
    for member in zip_ref.namelist():
        if not _is_safe_path(dest_dir, member):
            raise ValueError(f"Attempted path traversal in archive: {member}")
    zip_ref.extractall(dest_dir)


def _safe_extract_tar(tar_ref, dest_dir: str) -> None:
    """Safely extract tar archive with path traversal protection."""
    for member in tar_ref.getmembers():
        if not _is_safe_path(dest_dir, member.name):
            raise ValueError(f"Attempted path traversal in archive: {member.name}")
    tar_ref.extractall(dest_dir)


def _safe_extract_7z(archive, dest_dir: str) -> None:
    """Safely extract 7z archive with path traversal protection."""
    for name in archive.getnames():
        if not _is_safe_path(dest_dir, name):
            raise ValueError(f"Attempted path traversal in archive: {name}")
    archive.extractall(path=dest_dir)


def extract_archive(archive_file_path: str) -> str:
    """
    Extracts an archive to a temporary folder.

    Parameters
    ----------
    archive_file_path : str
        The path to the archive file.

    Returns
    -------
    str
        The path to the extracted folder.
    """

    import tempfile

    from litemind.utils.temp_file_manager import register_temp_dir

    temp_dir = register_temp_dir(tempfile.mkdtemp(prefix="extracted_"))

    try:

        # Normalise the URI to a local file path:
        archive_file_path = uri_to_local_file_path(archive_file_path)

        if archive_file_path.endswith(".zip"):
            try:
                import zipfile
            except ImportError:
                raise ImportError("zipfile module is not installed")
            with zipfile.ZipFile(archive_file_path, "r") as zip_ref:
                _safe_extract_zip(zip_ref, temp_dir)

        elif archive_file_path.endswith(".tar.gz") or archive_file_path.endswith(
            ".tgz"
        ):
            try:
                import tarfile
            except ImportError:
                raise ImportError("tarfile module is not installed")
            with tarfile.open(archive_file_path, "r:gz") as tar_ref:
                _safe_extract_tar(tar_ref, temp_dir)

        elif archive_file_path.endswith(".tar.bz2") or archive_file_path.endswith(
            ".tbz"
        ):
            try:
                import tarfile
            except ImportError:
                raise ImportError("tarfile module is not installed")
            with tarfile.open(archive_file_path, "r:bz2") as tar_ref:
                _safe_extract_tar(tar_ref, temp_dir)

        elif archive_file_path.endswith(".tar.xz") or archive_file_path.endswith(
            ".txz"
        ):
            try:
                import tarfile
            except ImportError:
                raise ImportError("tarfile module is not installed")
            with tarfile.open(archive_file_path, "r:xz") as tar_ref:
                _safe_extract_tar(tar_ref, temp_dir)

        elif archive_file_path.endswith(".7z"):
            try:
                import py7zr
            except ImportError:
                raise ImportError("py7zr module is not installed")
            with py7zr.SevenZipFile(archive_file_path, "r") as archive:
                _safe_extract_7z(archive, temp_dir)

        elif archive_file_path.endswith(".rar"):
            try:
                import patoolib
            except ImportError:
                raise ImportError("patoolib module is not installed")
            # Note: patoolib doesn't provide a way to list archive contents before extraction.
            # We rely on the underlying extraction tool (unrar/7z) to reject path traversal.
            # Modern versions of these tools refuse to extract files with ".." in paths.
            patoolib.extract_archive(archive_file_path, outdir=temp_dir)

        else:
            raise ValueError(f"Unsupported archive format: {archive_file_path}")

    except Exception as e:
        import shutil

        shutil.rmtree(temp_dir)
        raise e

    # delete the __MACOSX folder if it exists:
    macosx_folder = os.path.join(temp_dir, "__MACOSX")
    if os.path.exists(macosx_folder):
        import shutil

        shutil.rmtree(macosx_folder)

    return temp_dir


if __name__ == "__main__":
    archive_path = "your_archive.zip"  # Replace with the actual path
    extracted_dir = extract_archive(archive_path)
    print("Archive extracted to:", extracted_dir)
