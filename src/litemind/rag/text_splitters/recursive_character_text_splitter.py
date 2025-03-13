from typing import Optional, List

from litemind.rag.text_splitters.text_splitters_base import TextSplitter


class RecursiveCharacterTextSplitter(TextSplitter):
    """
    A text splitter that recursively splits text based on character count,
    using a list of separators.
    """

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            separators: Optional[List[str]] = None,
            character_splitting: bool = False,
    ):
        """
        Initialize the recursive character text splitter.

        Parameters
        ----------
        chunk_size: int
            The target size of each chunk.
        chunk_overlap: int
            The overlap between consecutive chunks.
        separators: Optional[List[str]]
            The separators to use when looking for chunk boundaries,
            in order of preference. Defaults to ["\n\n", "\n", ". ", " ", ""].
        character_splitting: bool
            Whether to split by characters if no separator is found.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.character_splitting = character_splitting

    def split_text(self, text: str) -> List[str]:
        """
        Split text recursively using the list of separators.

        Parameters
        ----------
        text: str
            The text to split.

        Returns
        -------
        List[str]
            The text chunks.
        """
        # If text fits in one chunk, return it as is
        if len(text) <= self.chunk_size:
            return [text]

        # Start the recursive splitting
        chunks = self._split_text_recursive(text, 0, "")

        # Filter out chunks that consist only of delimiters
        chunks = self._clean_chunks(chunks)

        return chunks

    def _split_text_recursive(self, text: str, separator_idx: int, previous_chunk: str) -> List[str]:
        """Split text recursively and handle overlaps using natural boundaries."""
        # Base case: text fits in a chunk
        if len(text) <= self.chunk_size:
            # If we need to add overlap with previous chunk
            if previous_chunk and self.chunk_overlap > 0:
                overlap = self._get_smart_overlap(previous_chunk, separator_idx)
                if not text.startswith(overlap):
                    text = overlap + text
            return [text]

        # If we've exhausted all separators, split by characters
        if separator_idx >= len(self.separators):
            if self.character_splitting:
                return self._split_by_chars(text, previous_chunk)
            else:
                return [text]

        # Get current separator
        separator = self.separators[separator_idx]

        # Empty separator means character-level splitting
        if self.character_splitting and separator == "":
            return self._split_by_chars(text, previous_chunk)

        # Split by current separator
        splits = text.split(separator)

        # If separator not found, try next separator
        if len(splits) == 1:
            return self._split_text_recursive(text, separator_idx + 1, previous_chunk)

        # Process the splits
        chunks = []
        current_chunk = ""

        for i, split in enumerate(splits):
            # Add separator back for all non-first splits
            if i > 0:
                split = separator + split

            # If adding this split would exceed chunk size
            if len(current_chunk) + len(split) > self.chunk_size:
                # Add current chunk to results if not empty
                if current_chunk:
                    # Add overlap from previous chunk if needed
                    if previous_chunk and not chunks and self.chunk_overlap > 0:
                        overlap = self._get_smart_overlap(previous_chunk, separator_idx)
                        if not current_chunk.startswith(overlap):
                            current_chunk = overlap + current_chunk
                    chunks.append(current_chunk)

                # Process the split
                if len(split) > self.chunk_size:
                    # Recursively process with next separator, passing current chunk for overlap
                    sub_chunks = self._split_text_recursive(split, separator_idx + 1,
                                                            current_chunk if current_chunk else previous_chunk)
                    chunks.extend(sub_chunks)
                else:
                    # Add overlap if needed
                    if current_chunk and self.chunk_overlap > 0:
                        overlap = self._get_smart_overlap(current_chunk, separator_idx)
                        if not split.startswith(overlap):
                            split = overlap + split
                    chunks.append(split)

                # Reset current chunk
                current_chunk = ""
            else:
                # Add split to current chunk
                current_chunk += split

        # Add final chunk if not empty
        if current_chunk:
            # Add overlap from previous chunk if needed
            if previous_chunk and not chunks and self.chunk_overlap > 0:
                overlap = self._get_smart_overlap(previous_chunk, separator_idx)
                if not current_chunk.startswith(overlap):
                    current_chunk = overlap + current_chunk
            chunks.append(current_chunk)

        return chunks

    def _get_smart_overlap(self, chunk: str, separator_idx: int) -> str:
        """Get overlap text using next level delimiter if possible."""

        # If there are no more separators, return the whole chunk
        if separator_idx >= len(self.separators) - 1:
            return chunk

        # Try to find a natural delimiter in the overlap region
        next_separator = self.separators[separator_idx + 1]
        if not next_separator:
            return chunk

        # Look for the last occurrence of next_separator within a reasonable region
        search_region = chunk[-min(self.chunk_overlap * 2, len(chunk)):]
        last_sep_pos = search_region.rfind(next_separator)

        if last_sep_pos >= 0:
            natural_overlap = search_region[last_sep_pos:]
            # Only use natural overlap if it doesn't exceed our desired overlap size
            if len(natural_overlap) <= self.chunk_overlap:
                return natural_overlap
            else:
                return chunk
        else:
            return chunk

    def _clean_chunks(self, chunks: List[str]) -> List[str]:
        """
        Clean up chunks by:
        1. Removing chunks that consist only of delimiters
        2. Trimming whitespace from the beginning and end of each chunk
        3. Removing duplicate chunks

        Parameters
        ----------
        chunks: List[str]
            List of text chunks

        Returns
        -------
        List[str]
            Cleaned chunks with duplicates removed
        """
        # If no chunks or no separators, return as is
        if not chunks or not self.separators:
            return chunks

        # Create a set of all characters that appear in any separator
        delimiter_chars = set()
        for separator in self.separators:
            for char in separator:
                delimiter_chars.add(char)

        result = []
        seen_chunks = set()  # To track duplicates

        for chunk in chunks:
            # Skip empty chunks
            if not chunk:
                continue

            # Strip whitespace from beginning and end
            chunk = chunk.strip()

            # Skip if chunk is now empty after stripping
            if not chunk:
                continue

            # Skip if chunk is all whitespace (should be caught by strip, but just in case)
            if chunk.isspace():
                continue

            # Check if the chunk consists entirely of delimiter characters
            is_delimiter_only = all(char in delimiter_chars for char in chunk)

            # Skip delimiter-only chunks
            if is_delimiter_only:
                continue

            # Skip duplicates
            if chunk in seen_chunks:
                continue

            # Add to results and mark as seen
            result.append(chunk)
            seen_chunks.add(chunk)

        return result

