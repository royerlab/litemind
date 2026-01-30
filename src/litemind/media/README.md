# Media Package

The `media` package implements a comprehensive multimodal media handling system for Litemind, supporting diverse content types with automatic format conversion capabilities.

## Package Structure

```
media/
├── media_base.py            # Abstract base interface
├── media_default.py         # Default implementation mixin
├── media_uri.py             # URI-based media abstraction
├── types/                   # Concrete media type implementations
│   ├── media_text.py        # Plain text
│   ├── media_code.py        # Source code with language
│   ├── media_json.py        # JSON structures
│   ├── media_image.py       # 2D images
│   ├── media_audio.py       # Audio files
│   ├── media_video.py       # Video files
│   ├── media_document.py    # Multipage documents (PDFs)
│   ├── media_table.py       # Tabular data
│   ├── media_ndimage.py     # N-dimensional images
│   ├── media_object.py      # Pydantic BaseModel objects
│   ├── media_file.py        # Generic files
│   ├── media_action.py      # Action descriptions
│   └── media_types.py       # Registry of all types
├── conversion/              # Media conversion pipeline
│   ├── media_converter.py   # Main conversion orchestrator
│   └── converters/          # Specialized converters
└── tests/                   # Test suite
```

## Base Classes

### MediaBase (`media_base.py`)

Abstract interface defining the contract for all media types:

```python
from abc import ABC, abstractmethod

class MediaBase(ABC):
    @abstractmethod
    def get_content(self) -> Any:
        """Get the content of the media."""
        pass

    @abstractmethod
    def to_message_block(self) -> MessageBlock:
        """Convert to message block for API transmission."""
        pass
```

### MediaDefault (`media_default.py`)

Default implementation mixin providing common functionality:
- `to_message_block()` - Creates MessageBlock with content
- `__str__()`, `__len__()`, `__contains__()`, `__hash__()`, `__eq__()`
- Pickle serialization support via `PickleSerializable`

### MediaURI (`media_uri.py`)

Abstraction for media accessible via URIs (local files, remote URLs, data URIs):

```python
from litemind.media.media_uri import MediaURI

media = MediaURI("https://example.com/image.png")
media.is_local()           # False
media.get_extension()      # ".png"
media.to_local_file_path() # Downloads and returns local path
media.to_base64_data()     # Base64 encoded content
```

## Media Types

### Text (`media_text.py`)

Plain text content:

```python
from litemind.media.types.media_text import Text

text = Text("Hello, world!")
text.get_content()  # "Hello, world!"
```

### Code (`media_code.py`)

Source code with language specification:

```python
from litemind.media.types.media_code import Code

code = Code("def hello(): return 'world'", lang="python")
code.to_markdown()  # ```python\ndef hello(): return 'world'\n```
```

### Json (`media_json.py`)

JSON structures:

```python
from litemind.media.types.media_json import Json

json_data = Json({"key": "value", "numbers": [1, 2, 3]})
json_data.get_content()  # dict
json_data.to_markdown_string()  # Formatted JSON code block
```

### Image (`media_image.py`)

2D images with PIL integration:

```python
from litemind.media.types.media_image import Image

img = Image("path/to/image.jpg")
img.open_pil_image()      # PIL.Image object
img.normalise_to_png()    # Returns new Image in PNG format
img.normalise_to_jpeg()   # Returns new Image in JPEG format

# From numpy array
img = Image.from_data(numpy_array, filepath="/tmp/output.png")

# From PIL image
img = Image.from_PIL_image(pil_image, filepath="/tmp/output.png")
```

### Audio (`media_audio.py`)

Audio files with soundfile integration:

```python
from litemind.media.types.media_audio import Audio

audio = Audio("path/to/audio.wav")
audio.get_info_markdown()  # Duration, sample rate, channels

# From numpy array
audio = Audio.from_data(
    data=numpy_array,
    sample_rate=16000,
    file_format="wav"
)
```

### Video (`media_video.py`)

Video files with FFmpeg integration:

```python
from litemind.media.types.media_video import Video

video = Video("path/to/video.mp4")
video.get_video_info()  # Duration, resolution, codec, frame rate

# Extract frames and audio
frames, audio = video.convert_to_frames_and_audio(
    frame_interval=1.0,  # 1 frame per second
    key_frames=False
)
```

### Document (`media_document.py`)

Multipage documents (PDFs):

```python
from litemind.media.types.media_document import Document

doc = Document("path/to/document.pdf")
pages_text = doc.extract_text_from_pages()  # List of text per page
page_images = doc.take_image_of_each_page(dpi=300)  # Rendered pages
```

### Table (`media_table.py`)

Tabular data with pandas integration:

```python
from litemind.media.types.media_table import Table

table = Table("path/to/data.csv")
df = table.to_dataframe()  # pandas DataFrame
table.to_markdown()  # Markdown table

# From DataFrame
table = Table.from_table(dataframe, filepath="/tmp/output.csv")

# From CSV string
table = Table.from_csv_string("col1,col2\n1,2\n3,4")
```

### NdImage (`media_ndimage.py`)

N-dimensional images (microscopy, scientific imaging):

```python
from litemind.media.types.media_ndimage import NdImage

nd_img = NdImage("path/to/stack.tif")  # Supports .npy, .npz, .tif

# Convert to 2D projections with description
text_and_images = nd_img.to_text_and_2d_projection_medias(
    channel_threshold=4  # Treat dims with <=4 elements as channels
)
```

### Object (`media_object.py`)

Pydantic BaseModel wrapper:

```python
from litemind.media.types.media_object import Object
from pydantic import BaseModel

class MyData(BaseModel):
    name: str
    value: int

obj = Object(MyData(name="test", value=42))
obj.to_json_string()  # '{"name": "test", "value": 42}'
obj.to_json_media()   # Json media object
```

### File (`media_file.py`)

Generic files (fallback for unrecognized types):

```python
from litemind.media.types.media_file import File

file = File("path/to/unknown.xyz")
file.to_markdown_text_media(hex_dump_length=64)  # File info + hex dump
```

## Media Conversion System

### MediaConverter (`conversion/media_converter.py`)

Orchestrates conversion of multimodal messages to specified allowed media types:

```python
from litemind.media.conversion.media_converter import MediaConverter

converter = MediaConverter()
converter.add_default_converters()

# Convert message blocks to allowed types
converted_message = message.convert_media(
    media_converter=converter,
    allowed_media_types={Text, Image}
)
```

### Available Converters

| Converter | Input | Output |
|-----------|-------|--------|
| CodeConverter | Code | Text (Markdown) |
| JsonConverter | Json | Text (Markdown) |
| TableConverter | Table | Text (Markdown) |
| ObjectConverter | Object | Text (Markdown JSON) |
| NdImageConverter | NdImage | Text + Image (projections) |
| AudioConverterWhisperLocal | Audio | Text (transcription) |
| VideoConverterFfmpeg | Video | Text + Image + Audio |
| DocumentConverterPymupdf | Document | Text + Image |
| DocumentConverterDocling | Document | Text + Image |
| FileConverter | File | Text (hex dump) |

### Custom Converters

Implement `BaseConverter` to add custom conversions:

```python
from litemind.media.conversion.converters.base_converter import BaseConverter

class MyConverter(BaseConverter):
    def rule(self):
        return {MyMediaType: {Text, Image}}

    def can_convert(self, media):
        return isinstance(media, MyMediaType)

    def convert(self, media):
        return [Text(str(media)), Image(media.thumbnail)]

converter.add_media_converter(MyConverter())
```

## Architecture

### Layered Design

1. **Abstract layer**: `MediaBase` defines the contract
2. **Default implementation**: `MediaDefault` provides common methods
3. **URI abstraction**: `MediaURI` handles remote/local access
4. **Concrete types**: 12 specialized media classes

### Conversion Pipeline

```
Input Media
     ↓
MediaConverter.can_convert_within()
     ↓
Priority-based converter selection
     ↓
Recursive conversion until convergence
     ↓
Allowed media types only
```

## Dependencies

- **Core**: NumPy, Pandas
- **Image**: PIL, tifffile, imageio
- **Audio**: soundfile, (optional) Whisper
- **Video**: FFmpeg
- **Documents**: PyMuPDF, Docling
- **Data**: pydantic

## Docstring Coverage

| Module | Coverage |
|--------|----------|
| media_base.py | 100% |
| media_uri.py | 85% |
| types/media_text.py | 100% |
| types/media_code.py | 100% |
| types/media_image.py | 80% |
| types/media_audio.py | 85% |
| types/media_video.py | 85% |
| types/media_document.py | 50% (needs improvement) |
| types/media_table.py | 75% |
| types/media_ndimage.py | 90% |
| conversion/media_converter.py | 95% |
| conversion/converters/ | 90% |

The package follows numpy-style docstrings. Priority improvements needed for `media_document.py`.
