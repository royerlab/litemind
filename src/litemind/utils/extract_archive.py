def extract_archive(archive_path: str) -> str:
    """
    Extracts an archive to a temporary folder.

    Parameters
    ----------
    archive_path : str
        The path to the archive file.

    Returns
    -------
    str
        The path to the extracted folder.
    """

    import tempfile
    temp_dir = tempfile.mkdtemp()

    try:

        # If the path is in URI format, and local, then convert it to a local path:
        if archive_path.startswith('file://'):
            # remove the 'file://' prefix
            archive_path = archive_path.replace('file://', '')

        # if the file is not local, download it to a temporary file:
        elif archive_path.startswith('http://') or archive_path.startswith('https://'):
            import requests
            import os
            import shutil
            import tempfile

            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.close()

            # Download the file
            response = requests.get(archive_path, stream=True)
            with open(temp_file.name, 'wb') as f:
                shutil.copyfileobj(response.raw, f)

            # Use the temporary file as the archive path
            archive_path = temp_file.name

        if archive_path.endswith('.zip'):
            try:
                import zipfile
            except ImportError:
                raise ImportError("zipfile module is not installed")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            try:
                import tarfile
            except ImportError:
                raise ImportError("tarfile module is not installed")
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(temp_dir)

        elif archive_path.endswith('.tar.bz2') or archive_path.endswith('.tbz'):
            try:
                import tarfile
            except ImportError:
                raise ImportError("tarfile module is not installed")
            with tarfile.open(archive_path, 'r:bz2') as tar_ref:
                tar_ref.extractall(temp_dir)

        elif archive_path.endswith('.tar.xz') or archive_path.endswith('.txz'):
            try:
                import tarfile
            except ImportError:
                raise ImportError("tarfile module is not installed")
            with tarfile.open(archive_path, 'r:xz') as tar_ref:
                tar_ref.extractall(temp_dir)

        elif archive_path.endswith('.7z'):
            try:
                import py7zr
            except ImportError:
                raise ImportError("py7zr module is not installed")
            with py7zr.SevenZipFile(archive_path, 'r') as archive:
                archive.extractall(path=temp_dir)

        elif archive_path.endswith('.rar'):
            try:
                import patoolib
            except ImportError:
                raise ImportError("patoolib module is not installed")
            patoolib.extract_archive(archive_path, outdir=temp_dir)

        else:
            raise ValueError("Unsupported archive format")


    except Exception as e:
        import shutil
        shutil.rmtree(temp_dir)
        raise e

    return temp_dir


if __name__ == "__main__":
    archive_path = "your_archive.zip"  # Replace with the actual path
    extracted_dir = extract_archive(archive_path)
    print("Archive extracted to:", extracted_dir)
