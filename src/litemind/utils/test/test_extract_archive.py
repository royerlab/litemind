import os
import shutil
import tempfile

import pytest

from litemind.utils.extract_archive import extract_archive


@pytest.fixture
def create_temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_extract_zip_archive(create_temp_dir):
    import zipfile
    archive_path = os.path.join(create_temp_dir, 'test.zip')
    with zipfile.ZipFile(archive_path, 'w') as zipf:
        zipf.writestr('test.txt', 'This is a test file')

    extracted_dir = extract_archive(archive_path)
    assert os.path.exists(extracted_dir)
    assert os.path.isfile(os.path.join(extracted_dir, 'test.txt'))


def test_extract_tar_gz_archive(create_temp_dir):
    import tarfile
    archive_path = os.path.join(create_temp_dir, 'test.tar.gz')
    with tarfile.open(archive_path, 'w:gz') as tarf:
        tarf.addfile(tarfile.TarInfo('test.txt'), open(__file__, 'rb'))

    extracted_dir = extract_archive(archive_path)
    assert os.path.exists(extracted_dir)
    assert os.path.isfile(os.path.join(extracted_dir, 'test.txt'))


def test_extract_tar_bz2_archive(create_temp_dir):
    import tarfile
    archive_path = os.path.join(create_temp_dir, 'test.tar.bz2')
    with tarfile.open(archive_path, 'w:bz2') as tarf:
        tarf.addfile(tarfile.TarInfo('test.txt'), open(__file__, 'rb'))

    extracted_dir = extract_archive(archive_path)
    assert os.path.exists(extracted_dir)
    assert os.path.isfile(os.path.join(extracted_dir, 'test.txt'))


def test_extract_tar_xz_archive(create_temp_dir):
    import tarfile
    archive_path = os.path.join(create_temp_dir, 'test.tar.xz')
    with tarfile.open(archive_path, 'w:xz') as tarf:
        tarf.addfile(tarfile.TarInfo('test.txt'), open(__file__, 'rb'))

    extracted_dir = extract_archive(archive_path)
    assert os.path.exists(extracted_dir)
    assert os.path.isfile(os.path.join(extracted_dir, 'test.txt'))


def test_extract_7z_archive(create_temp_dir):
    try:
        import py7zr
    except ImportError:
        pytest.skip("py7zr module is not installed")

    import py7zr
    archive_path = os.path.join(create_temp_dir, 'test.7z')
    with py7zr.SevenZipFile(archive_path, 'w') as archive:
        archive.writeall(__file__, 'test.txt')

    extracted_dir = extract_archive(archive_path)
    assert os.path.exists(extracted_dir)
    assert os.path.isfile(os.path.join(extracted_dir, 'test.txt'))


def test_extract_rar_archive(create_temp_dir):
    try:
        import patoolib
    except ImportError:
        pytest.skip("patoolib module is not installed")

    import patoolib
    archive_path = os.path.join(create_temp_dir, 'test.rar')
    patoolib.create_archive(archive_path, [__file__])

    extracted_dir = extract_archive(archive_path)
    assert os.path.exists(extracted_dir)
    assert os.path.isfile(os.path.join(extracted_dir, 'test.txt'))
