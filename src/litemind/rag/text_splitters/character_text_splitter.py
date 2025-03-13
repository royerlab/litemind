from typing import List

from litemind.rag.text_splitters.text_splitters_base import TextSplitter


class CharacterTextSplitter(TextSplitter):
    """
    A text splitter that splits text based on character count.
    """

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            separator: str = "\n",
    ):
        """
        Initialize the character text splitter.

        Parameters
        ----------
        chunk_size: int
            The target size of each chunk.
        chunk_overlap: int
            The overlap between consecutive chunks.
        separator: str
            The separator to use when looking for chunk boundaries.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on character count.

        Parameters
        ----------
        text: str
            The text to split.

        Returns
        -------
        List[str]
            The text chunks.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to find a natural split point
            if end < len(text):
                # First try to split at the separator
                separator_pos = text.rfind(self.separator, start, end)
                if separator_pos > start:
                    end = separator_pos + len(self.separator)
                else:
                    # Fall back to splitting at the character level
                    # Try to split at spaces or punctuation
                    for separator in ['. ', '! ', '? ', '.\n', '!\n', '?\n', '\n\n', '\n', ' ']:
                        separator_pos = text.rfind(separator, start, end)
                        if separator_pos > start:
                            end = separator_pos + len(separator)
                            break

            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return chunks