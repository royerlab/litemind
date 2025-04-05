from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


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

    temp_dir = tempfile.mkdtemp(prefix="extracted_")

    try:

        # Normalise the URI to a local file path:
        archive_file_path = uri_to_local_file_path(archive_file_path)

        if archive_file_path.endswith(".zip"):
            try:
                import zipfile
            except ImportError:
                raise ImportError("zipfile module is not installed")
            with zipfile.ZipFile(archive_file_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

        elif archive_file_path.endswith(".tar.gz") or archive_file_path.endswith(
            ".tgz"
        ):
            try:
                import tarfile
            except ImportError:
                raise ImportError("tarfile module is not installed")
            with tarfile.open(archive_file_path, "r:gz") as tar_ref:
                tar_ref.extractall(temp_dir)

        elif archive_file_path.endswith(".tar.bz2") or archive_file_path.endswith(
            ".tbz"
        ):
            try:
                import tarfile
            except ImportError:
                raise ImportError("tarfile module is not installed")
            with tarfile.open(archive_file_path, "r:bz2") as tar_ref:
                tar_ref.extractall(temp_dir)

        elif archive_file_path.endswith(".tar.xz") or archive_file_path.endswith(
            ".txz"
        ):
            try:
                import tarfile
            except ImportError:
                raise ImportError("tarfile module is not installed")
            with tarfile.open(archive_file_path, "r:xz") as tar_ref:
                tar_ref.extractall(temp_dir)

        elif archive_file_path.endswith(".7z"):
            try:
                import py7zr
            except ImportError:
                raise ImportError("py7zr module is not installed")
            with py7zr.SevenZipFile(archive_file_path, "r") as archive:
                archive.extractall(path=temp_dir)

        elif archive_file_path.endswith(".rar"):
            try:
                import patoolib
            except ImportError:
                raise ImportError("patoolib module is not installed")
            patoolib.extract_archive(archive_file_path, outdir=temp_dir)

        else:
            raise ValueError(f"Unsupported archive format: {archive_file_path}")

    except Exception as e:
        import shutil

        shutil.rmtree(temp_dir)
        raise e

    # delete the __MACOSX folder if it exists:
    import os

    macosx_folder = os.path.join(temp_dir, "__MACOSX")
    if os.path.exists(macosx_folder):
        import shutil

        shutil.rmtree(macosx_folder)

    return temp_dir


if __name__ == "__main__":
    archive_path = "your_archive.zip"  # Replace with the actual path
    extracted_dir = extract_archive(archive_path)
    print("Archive extracted to:", extracted_dir)
