from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_document import Document
from litemind.media.types.media_text import Text
from litemind.utils.file_types.file_types import classify_uri
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class DocumentConverterPythonMinify(BaseConverter):
    """Converts Python source files to Text after minification.

    Uses ``python_minifier`` to reduce code size before wrapping in a
    Markdown fenced code block. This is useful for reducing token usage
    when sending source code to LLMs.
    """

    def __init__(
        self,
        remove_pass: bool = True,
        remove_literal_statements: bool = True,
        combine_imports: bool = True,
        hoist_literals: bool = True,
        rename_locals: bool = True,
        preserve_locals: bool = None,
        rename_globals: bool = False,
        preserve_globals: bool = None,
        remove_object_base: bool = True,
        convert_posargs_to_args: bool = True,
        preserve_shebang: bool = True,
        remove_asserts: bool = False,
        remove_debug: bool = False,
        remove_explicit_return_none: bool = True,
        remove_builtin_exception_brackets: bool = True,
        constant_folding: bool = True,
    ):
        """Initialise the converter with minification options.

        All parameters correspond to ``python_minifier.minify()`` arguments.

        Parameters
        ----------
        remove_pass : bool
            Remove ``pass`` statements.
        remove_literal_statements : bool
            Remove standalone literal expressions (e.g. docstrings).
        combine_imports : bool
            Merge multiple import statements.
        hoist_literals : bool
            Hoist repeated literals to module-level variables.
        rename_locals : bool
            Shorten local variable names.
        preserve_locals : list or None
            Local names to keep unchanged.
        rename_globals : bool
            Shorten global variable names.
        preserve_globals : list or None
            Global names to keep unchanged.
        remove_object_base : bool
            Remove explicit ``object`` base class.
        convert_posargs_to_args : bool
            Convert positional-only args to regular args.
        preserve_shebang : bool
            Keep the shebang line.
        remove_asserts : bool
            Remove ``assert`` statements.
        remove_debug : bool
            Remove ``__debug__``-guarded code.
        remove_explicit_return_none : bool
            Remove ``return None`` at function ends.
        remove_builtin_exception_brackets : bool
            Remove empty parentheses from exception constructors.
        constant_folding : bool
            Evaluate constant expressions at minify time.
        """
        super().__init__()
        self.kwargs_minify = {
            "remove_pass": remove_pass,
            "remove_literal_statements": remove_literal_statements,
            "combine_imports": combine_imports,
            "hoist_literals": hoist_literals,
            "rename_locals": rename_locals,
            "preserve_locals": preserve_locals if preserve_locals is not None else [],
            "rename_globals": rename_globals,
            "preserve_globals": (
                preserve_globals if preserve_globals is not None else []
            ),
            "remove_object_base": remove_object_base,
            "convert_posargs_to_args": convert_posargs_to_args,
            "preserve_shebang": preserve_shebang,
            "remove_asserts": remove_asserts,
            "remove_debug": remove_debug,
            "remove_explicit_return_none": remove_explicit_return_none,
            "remove_builtin_exception_brackets": remove_builtin_exception_brackets,
            "constant_folding": constant_folding,
        }

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        return [(Document, [Text])]

    def can_convert(self, media: MediaBase) -> bool:

        # Check is the media is None:
        if media is None:
            return False

        # Check if the media is a Document:
        if not isinstance(media, Document):
            return False

        # Check if the file is Python code:
        file_type = classify_uri(media.uri)
        if file_type not in {"code"} and not media.get_extension() in {"py"}:
            return False

        # By default, we can convert the media, unless the previous checks failed:
        return True

    def convert(self, media: MediaBase) -> List[MediaBase]:

        # Check if the media is a Document:
        if not isinstance(media, Document):
            raise ValueError(f"Expected Document media, got {type(media)}")

        # get filename:
        filename = media.get_filename()

        # Get the file extension:
        extension = media.get_extension()

        # Check that the file is a Python file:
        if media.get_extension() != "py":
            raise ValueError(f"Expected a Python file, got {media.get_extension()}")

        # Normalise uri to local path:
        document_path = uri_to_local_file_path(media.uri)

        # Read the content of the text document from file:
        with open(document_path, "r", encoding="utf-8") as file:
            original_code = file.read()

        from python_minifier import minify

        minified_code = minify(original_code, **self.kwargs_minify)

        # Wrap file in markdown quotes and give filename:
        text_content = f"{filename}\n```{extension}\n{minified_code}\n```"

        # Create a Text media object:
        text_media = Text(text_content)

        # Return the converted media:
        return [text_media]
