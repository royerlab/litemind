from litemind.apis.utils.dowload_video_to_tempfile import \
    download_video_to_temp_file
from litemind.apis.utils.sample_video import \
    append_video_frames_and_audio_to_message
from litemind.apis.utils.write_base64_to_temp_file import \
    write_base64_to_temp_file


def transform_video_uris_to_images_and_video(message):
    # Add and convert each video URI to the message as images and audio:
    for video_uri in message.video_uris:

        if video_uri.startswith("data:video/"):
            local_path = write_base64_to_temp_file(video_uri)

        elif video_uri.startswith("http://") or video_uri.startswith(
                "https://"):
            local_path = download_video_to_temp_file(video_uri)

        elif video_uri.startswith("file://"):
            local_path = video_uri.replace("file://", "")

        else:
            raise ValueError(
                f"Invalid video URI: '{video_uri}' (must start with 'data:video/', 'http://', 'https://', or 'file://')")

        message = append_video_frames_and_audio_to_message(
            local_path, message)
    return message
