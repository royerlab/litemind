"""Registry of all concrete media type classes.

Provides a single function to lazily import and return every media type,
avoiding circular import issues at module load time.
"""

from typing import List, Type


def all_media_types() -> List[Type["MediaBase"]]:
    """Return a list of all concrete media type classes.

    This function lazily imports every media type to avoid circular import
    issues. Useful for iteration over the full set of media types, e.g.
    when building a conversion graph.

    Returns
    -------
    List[Type[MediaBase]]
        All registered media type classes in alphabetical order.
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
