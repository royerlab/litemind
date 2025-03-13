from typing import Optional

from litemind.rag.text_splitters.recursive_character_text_splitter import RecursiveCharacterTextSplitter


class CodeTextSplitter(RecursiveCharacterTextSplitter):
    """
    A text splitter optimized for code.
    """

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            language: Optional[str] = None,
    ):
        """
        Initialize the code text splitter.

        Parameters
        ----------
        chunk_size: int
            The target size of each chunk.
        chunk_overlap: int
            The overlap between consecutive chunks.
        language: Optional[str]
            The programming language of the code.
        """
        # Common separators for code
        separators = [
            "\nclass ",
            "\ndef ",
            "\nfunction ",
            "\nif ",
            "\nfor ",
            "\nwhile ",
            "\n\n",
            "\n",
            " ",
            "",
        ]

        # Add language-specific separators
        if language:
            language = language.lower()
            if language in ["python", "py"]:
                separators = [
                    "\nclass ",
                    "\ndef ",
                    "\n    def ",
                    "\n\n",
                    "\n",
                    " ",
                    "",
                ]
            elif language in ["javascript", "js", "typescript", "ts"]:
                separators = [
                    "\nfunction ",
                    "\nconst ",
                    "\nlet ",
                    "\nvar ",
                    "\nclass ",
                    "\n\n",
                    "\n",
                    " ",
                    "",
                ]
            elif language in ["java", "c", "cpp", "c++", "csharp", "c#"]:
                separators = [
                    "\npublic class ",
                    "\nclass ",
                    "\nprivate class ",
                    "\nprotected class ",
                    "\npublic static ",
                    "\nprivate static ",
                    "\nprotected static ",
                    "\npublic ",
                    "\nprivate ",
                    "\nprotected ",
                    "\n\n",
                    "\n",
                    " ",
                    "",
                ]

        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
