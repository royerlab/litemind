from litemind.rag.text_splitters.recursive_character_text_splitter import RecursiveCharacterTextSplitter


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """
    A text splitter optimized for Markdown documents.
    """

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
    ):
        """
        Initialize the Markdown text splitter.

        Parameters
        ----------
        chunk_size: int
            The target size of each chunk.
        chunk_overlap: int
            The overlap between consecutive chunks.
        """
        separators = [
            # Headers
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n##### ",
            "\n###### ",
            "\n# ",
            # Lists and code blocks
            "\n\n```",
            "\n```",
            "\n\n- ",
            "\n\n* ",
            "\n\n+ ",
            "\n- ",
            "\n* ",
            "\n+ ",
            # Paragraphs and other elements
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            " ",
            "",
        ]
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
