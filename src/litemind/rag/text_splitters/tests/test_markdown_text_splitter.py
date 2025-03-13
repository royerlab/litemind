import pytest
from litemind.rag.text_splitters.markdown_text_splitter import MarkdownTextSplitter

def test_markdown_splitter_complex_document():
    splitter = MarkdownTextSplitter(chunk_size=30, chunk_overlap=5)
    text = """
# Header 1

This is a paragraph under header 1. It contains multiple sentences. Here's another sentence.

## Header 2

- List item 1
- List item 2
- List item 3

### Header 3
    
Another paragraph with more text. This paragraph is meant to test the splitting on sentences. Sentence 1. Sentence 2! Sentence 3?

#### Header 4

1. Ordered list item 1
2. Ordered list item 2
3. Ordered list item 3

##### Header 5

> Blockquote with some text.

###### Header 6

Final paragraph to test the end of the document. This should be the last chunk.
"""

    chunks = splitter.split_text(text)

    # Verify that the chunks are correctly split and overlap is applied
    assert len(chunks) > 1  # Ensure that the text is split into multiple chunks
    for chunk in chunks:
        assert len(chunk) <= 100 + 20  # Ensure each chunk is within the size limit including overlap

    # Check that specific parts of the text are present in the chunks
    assert any("# Header 1" in chunk for chunk in chunks)
    assert any("## Header 2" in chunk for chunk in chunks)
    assert any("### Header 3" in chunk for chunk in chunks)
    assert any("#### Header 4" in chunk for chunk in chunks)
    assert any("##### Header 5" in chunk for chunk in chunks)
    assert any("###### Header 6" in chunk for chunk in chunks)
    assert any("code block" in chunk for chunk in chunks)
    assert any("List item 1" in chunk for chunk in chunks)
    assert any("Ordered list item 1" in chunk for chunk in chunks)
    assert any("Blockquote with some text." in chunk for chunk in chunks)
    assert any("Final paragraph to test the end of the document." in chunk for chunk in chunks)

    # Ensure that the chunks cover the entire text
    reconstructed_text = ''.join(chunks).replace('\n', '').replace(' ', '')
    original_text = text.replace('\n', '').replace(' ', '')
    assert reconstructed_text == original_text