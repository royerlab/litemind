from typing import List, Type


def all_media_types() -> List[Type["MediaBase"]]:
    """
    Returns a list of all media types available in the litemind.media.types module.
    """

    from litemind.media.types.media_action import Action
    from litemind.media.types.media_audio import Audio
    from litemind.media.types.media_code import Code
    from litemind.media.types.media_document import Document
    from litemind.media.types.media_file import File
    from litemind.media.types.media_image import Image
    from litemind.media.types.media_json import Json
    from litemind.media.types.media_ndimage import NdImage
    from litemind.media.types.media_object import Object
    from litemind.media.types.media_table import Table
    from litemind.media.types.media_text import Text
    from litemind.media.types.media_video import Video

    # Rename to make it public
    media_types_list: List[Type["MediaBase"]] = [
        Action,
        Audio,
        Code,
        Document,
        File,
        Image,
        Json,
        NdImage,
        Object,
        Table,
        Text,
        Video,
    ]

    return media_types_list
