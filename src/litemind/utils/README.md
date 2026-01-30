# Utils Package

The `utils` package provides utility functions for multimedia processing, data serialization, file type detection, and cross-platform file handling.

## Package Structure

```
utils/
├── Image/Media Conversion
│   ├── convert_image_to_jpg.py
│   ├── convert_image_to_png.py
│   └── transform_video_uris_to_images_and_audio.py
├── Document Processing
│   ├── document_processing.py
│   └── markdown.py
├── Archive Handling
│   └── extract_archive.py
├── Embeddings
│   ├── fastembed_embeddings.py
│   └── random_projector.py
├── File Operations
│   ├── folder_description.py
│   ├── normalise_uri_to_local_file_path.py
│   ├── read_file_and_convert_to_base64.py
│   ├── temp_file_manager.py
│   └── uri_utils.py
├── Video/Audio
│   ├── ffmpeg_utils.py
│   ├── ffmpeg.py
│   └── whisper_transcribe_audio.py
├── JSON/Serialization
│   ├── json_utils.py
│   ├── json_to_object.py
│   ├── parse_json_output.py
│   ├── pickle_serialisation.py
│   └── recursive_dict_to_attributes.py
├── Data Processing
│   ├── load_table.py
│   ├── text_compressor.py
│   └── extract_thinking.py
├── Network
│   ├── free_port.py
│   └── get_media_type_from_uri.py
└── file_types/              # File type detection
    ├── file_extensions.py   # Extension sets by category
    └── file_types.py        # MIME detection and classification
```

## Key Utilities

### Image Conversion

```python
from litemind.utils.convert_image_to_jpg import convert_image_to_jpeg
from litemind.utils.convert_image_to_png import convert_image_to_png

jpeg_path = convert_image_to_jpeg("input.webp")  # Converts any format to JPEG
png_path = convert_image_to_png("input.jpg")     # Converts any format to PNG
```

### Document Processing

```python
from litemind.utils.document_processing import (
    convert_document_to_markdown,
    extract_text_from_document_pages,
    take_images_of_each_document_page,
    extract_images_from_document
)

# Convert PDF to Markdown
markdown = convert_document_to_markdown("document.pdf")

# Extract text per page
pages = extract_text_from_document_pages("document.pdf")

# Render pages as images
images = take_images_of_each_document_page("document.pdf", dpi=300)
```

### Archive Extraction

```python
from litemind.utils.extract_archive import extract_archive

# Supports: .zip, .tar.gz, .tgz, .tar.bz2, .7z, .rar
extracted_path = extract_archive("archive.zip")
```

Includes path traversal attack protection.

### Embeddings

```python
from litemind.utils.fastembed_embeddings import fastembed_text

embeddings = fastembed_text(
    texts=["Hello world", "How are you?"],
    model_name="BAAI/bge-large-en-v1.5",
    dimensions=512  # Optional dimensionality reduction
)
```

### URI Normalization

```python
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path

# Handles: local paths, file://, http(s)://, data: URIs, raw Base64
local_path = uri_to_local_file_path("https://example.com/image.png")
local_path = uri_to_local_file_path("data:image/png;base64,iVBORw...")
local_path = uri_to_local_file_path("file:///path/to/file.txt")
```

### FFmpeg Utilities

```python
from litemind.utils.ffmpeg_utils import (
    is_ffmpeg_available,
    get_video_info,
    extract_frames_and_audio,
    load_video_as_array
)

if is_ffmpeg_available():
    info = get_video_info("video.mp4")
    # {'duration': 120.5, 'width': 1920, 'height': 1080, 'fps': 30.0}

    extract_frames_and_audio(
        "video.mp4",
        output_dir="/tmp/frames",
        fps=1.0,  # 1 frame per second
        audio_sample_rate=16000
    )
```

### Audio Transcription

```python
from litemind.utils.whisper_transcribe_audio import (
    is_local_whisper_available,
    transcribe_audio_with_local_whisper
)

if is_local_whisper_available():
    text = transcribe_audio_with_local_whisper(
        "audio.wav",
        model_name="turbo"  # or "base", "small", "medium", "large"
    )
```

### Text Compression

```python
from litemind.utils.text_compressor import TextCompressor

compressor = TextCompressor(
    schemes=["newlines", "comments", "repeats", "trailing"]
)

compressed = compressor.compress(long_text)
```

Available schemes:
- `"newlines"` - Compress multiple newlines
- `"comments"` - Remove Python `#`, C `//`, SQL `--` comments
- `"repeats"` - Truncate repeated character sequences
- `"spaces"` - Convert 4 spaces to tabs
- `"trailing"` - Strip trailing whitespace
- `"imports"` - Prune Python import blocks

### JSON Utilities

```python
from litemind.utils.json_utils import JSONSerializable
from litemind.utils.parse_json_output import parse_json, extract_json_substring

# Mixin for automatic serialization
class MyClass(JSONSerializable):
    def __init__(self, name: str):
        self.name = name

obj = MyClass("test")
json_str = obj.to_json()
restored = MyClass.from_json(json_str)

# Parse JSON from LLM output
json_str = extract_json_substring("Here's the data: {\"key\": \"value\"}")
parsed = parse_json(json_str, MyPydanticModel)
```

### Pickle Serialization

```python
from litemind.utils.pickle_serialisation import PickleSerializable

class MyClass(PickleSerializable):
    pass

obj = MyClass()
pickled = obj.to_pickle()
restored = MyClass.from_pickle(pickled)

# Base64 encoding
encoded = obj.to_base64()
restored = MyClass.from_base64(encoded)
```

### Temporary File Management

```python
from litemind.utils.temp_file_manager import register_temp_file, register_temp_dir

# Register for automatic cleanup at process exit
temp_path = register_temp_file("/tmp/my_temp.txt")
temp_dir = register_temp_dir("/tmp/my_temp_dir")
```

### File Type Detection

```python
from litemind.utils.file_types.file_types import (
    classify,
    is_image_file,
    is_audio_file,
    is_video_file,
    is_document_file,
    probe
)

# Classify file type
file_type = classify("path/to/file.pdf")
# Returns: 'text', 'code', 'pdf', 'image', 'audio', 'video',
#          'archive', 'office', 'executable', 'web', 'script', 'binary'

# Check specific types
is_image_file("photo.jpg")    # True
is_audio_file("song.mp3")     # True
is_video_file("movie.mp4")    # True

# Deep probe with magic bytes
result = probe("unknown_file")
# {'extension_mime': 'image/png', 'signature_mime': 'image/png',
#  'libmagic_mime': 'image/png', 'is_text': False}
```

### Extension Categories

```python
from litemind.utils.file_types.file_extensions import (
    IMAGE_EXTS,      # 120+ image formats
    AUDIO_EXTS,      # 130+ audio formats
    VIDEO_EXTS,      # 100+ video formats
    DOCUMENT_EXTS,   # 50+ document formats
    ARCHIVE_EXTS,    # 80+ archive formats
    PROG_LANG_EXTS,  # 150+ programming languages
    TABLE_EXTS,      # CSV, Excel, databases
    OFFICE_EXTS,     # MS Office, OpenDocument
)
```

### Folder Description

```python
from litemind.utils.folder_description import (
    generate_tree_structure,
    human_readable_size,
    file_info_header
)

tree = generate_tree_structure(
    "/path/to/folder",
    allowed_extensions=[".py", ".md"],
    excluded_files=["__pycache__"],
    depth=3
)
```

## Design Patterns

- **LRU-cached availability checks** for optional dependencies
- **Automatic temporary file cleanup** via `atexit` hook
- **URI normalization** with fallback handling
- **Multi-layer MIME detection** (extension, magic bytes, libmagic)
- **Security-focused** path traversal protection in archive extraction

## Docstring Coverage

All utility functions have numpy-style docstrings with Parameters, Returns, and detailed descriptions. Coverage is approximately 95% across the package.
