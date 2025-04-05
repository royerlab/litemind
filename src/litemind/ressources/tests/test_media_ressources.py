import os

from litemind.ressources.media_resources import MediaResources


def test_get_local_test_folder_path_images():
    folder_path = MediaResources.get_local_test_folder_path("images")
    assert os.path.isdir(folder_path)


def test_get_local_test_file_uri_images():
    file_uri = MediaResources.get_local_test_file_uri("images", "cat.jpg")
    assert file_uri.startswith("file://")
    assert os.path.exists(file_uri.replace("file://", ""))


def test_get_local_test_image_uri():
    image_uri = MediaResources.get_local_test_image_uri("cat.jpg")
    assert image_uri.startswith("file://")
    assert os.path.exists(image_uri.replace("file://", ""))


def test_get_local_test_audio_uri():
    audio_uri = MediaResources.get_local_test_audio_uri("harvard.wav")
    assert audio_uri.startswith("file://")
    assert os.path.exists(audio_uri.replace("file://", ""))


def test_get_local_test_video_uri():
    video_uri = MediaResources.get_local_test_video_uri("bunny.mp4")
    assert video_uri.startswith("file://")
    assert os.path.exists(video_uri.replace("file://", ""))


def test_get_local_test_document_uri():
    doc_uri = MediaResources.get_local_test_document_uri("timaeus.txt")
    assert doc_uri.startswith("file://")
    assert os.path.exists(doc_uri.replace("file://", ""))


def test_get_local_test_table_uri():
    table_uri = MediaResources.get_local_test_table_uri("spreadsheet.csv")
    assert table_uri.startswith("file://")
    assert os.path.exists(table_uri.replace("file://", ""))


def test_get_local_test_archive_uri():
    archive_uri = MediaResources.get_local_test_archive_uri("alexander.zip")
    assert archive_uri.startswith("file://")
    assert os.path.exists(archive_uri.replace("file://", ""))
