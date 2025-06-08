"""text_compressor.py – extended text compressor

A lightweight text compressor that implements several safe compression schemes
to reduce text size while preserving information.
"""

import re
from textwrap import shorten
from typing import Dict, Iterable

IMPORT_RE = re.compile(
    r"""^\s*          # optional leading whitespace
        (?:from\s+\S+\s+import|import)\s+  # “import …” or “from … import …”
        [\s\S]*?       # everything up to line-end, including continuations
        (?:\\\s*)?$    # optional trailing back-slash for continuations
    """,
    re.VERBOSE,
)


class TextCompressor:
    """Text compressor with multiple safe compression schemes.

    Parameters
    ----------
    schemes
        Iterable of scheme-IDs to activate
    max_repeats
        Maximum number of times a character can be repeated (for "repeats" scheme)
    """

    _ALL_SCHEMES = (
        "newlines",  # Compress multiple newlines to a single one
        "comments",  # Remove single-line comments in code
        "repeats",  # Truncate repeated character sequences
        "spaces",  # Convert 4 spaces to tabs
        "trailing",  # Remove trailing whitespace
        "imports",  # Prune contiguous import blocks in Python code
    )

    def __init__(
        self,
        schemes: Iterable[str] = (
            "newlines",
            "comments",
            "repeats",
            "trailing",
            "imports",
        ),
        *,
        max_repeats: int = 16,
    ) -> None:
        unknown = set(schemes) - set(self._ALL_SCHEMES)
        if unknown:
            raise ValueError(f"Unknown scheme IDs: {', '.join(unknown)}")

        self.active: Dict[str, bool] = {s: (s in schemes) for s in self._ALL_SCHEMES}
        self.max_repeats = max_repeats

    def enable(self, *scheme_ids: str) -> None:
        """Enable specific compression schemes."""
        for s in scheme_ids:
            if s in self._ALL_SCHEMES:
                self.active[s] = True

    def disable(self, *scheme_ids: str) -> None:
        """Disable specific compression schemes."""
        for s in scheme_ids:
            if s in self._ALL_SCHEMES:
                self.active[s] = False

    def compress(self, text: str) -> str:
        """Return compressed text according to the active schemes."""
        if not text:
            return text

        # Remove trailing whitespace from each line
        if self.active.get("trailing"):
            lines = text.split("\n")
            lines = [line.rstrip() for line in lines]
            text = "\n".join(lines)

        # Compress multiple newlines to single newline
        if self.active.get("newlines"):
            text = re.sub(r"\n{2,}", "\n", text)

        # Handle code comments for various languages
        if self.active.get("comments"):
            # Split by newlines to process line by line
            lines = text.split("\n")
            result_lines = []

            for line in lines:
                # Match and remove common single-line comment styles
                # Python, Bash, Ruby, etc.
                line = re.sub(r"^\s*#.*$", "", line)
                # C, C++, Java, JavaScript, etc.
                line = re.sub(r"^\s*//.*$", "", line)
                # SQL
                line = re.sub(r"^\s*--.*$", "", line)
                result_lines.append(line)

            text = "\n".join(result_lines)

        # Truncate repeated character sequences
        if self.active.get("repeats"):

            def replace_repeats(match):
                char = match.group(1)
                return char * self.max_repeats

            text = re.sub(
                r"(.)\1{" + str(self.max_repeats) + ",}", replace_repeats, text
            )

        # Convert sequences of 4 spaces to tabs
        if self.active.get("spaces"):
            text = re.sub(r"    ", "\t", text)

        if self.active.get("imports"):
            text = self.prune_import_blocks(text)

        return text

    def prune_import_blocks(
        self,
        code: str,
        comment_template: str = "# [shortened] removed imports for {names}",
    ):
        """
        Replace *contiguous blocks* of import statements with a single comment.

        Parameters
        ----------
        code : str
            The original Python source.
        comment_template : str, optional
            Template for the replacement line.  It receives one
            argument, `names`, which is a comma-separated string of
            the imported symbols/modules found in the block.

        Returns
        -------
        str
            The modified source with import blocks replaced.
        """
        lines = code.splitlines()
        out, block, collected = [], [], []

        def flush_block():
            if not block:
                return
            # Concatenate all names we saw in this block
            flat = ", ".join(collected)
            # Trim the comment if it gets too long (tokens!)
            placeholder = shorten(
                comment_template.format(names=flat), width=140, placeholder=" …]"
            )
            out.append(placeholder)
            block.clear()
            collected.clear()

        for ln in lines:
            if IMPORT_RE.match(ln):
                block.append(ln)
                # Pull out the part after "import" (handles both syntaxes)
                after_import = re.split(r"\bimport\b", ln, maxsplit=1)[1]
                # Remove parentheses, back-slashes, comments
                cleaned = re.sub(r"[()\\#].*", "", after_import).strip()
                # Split on commas and "as" aliases
                for name in re.split(r",", cleaned):
                    n = name.strip().split(" as ")[0]
                    if n:
                        collected.append(n)
            else:
                flush_block()
                out.append(ln)
        flush_block()  # in case file ended with a block
        return "\n".join(out)
