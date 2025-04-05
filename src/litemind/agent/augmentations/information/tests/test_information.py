import uuid
from typing import Any, List

import numpy as np
import pytest

from litemind.agent.augmentations.information.information import Information
from litemind.media.media_default import MediaDefault


class MockMedia(MediaDefault):

    def __init__(self, content: Any):
        self._content = content

    def get_content(self) -> Any:
        return self._content

    def __str__(self) -> str:
        return str(self._content)


def simple_embedding_function(text: str) -> List[float]:
    return [float(len(text)), 0.5, 0.25]


def simple_summarization_function(content: str) -> str:
    words = content.split()
    return words[0] + "..." if len(words) > 1 else content


def test_init_with_all_parameters():
    media = MockMedia("Test content")
    info_id = str(uuid.uuid4())
    info = Information(
        media=media,
        summary="Test summary",
        metadata={"test_key": "test_value"},
        id=info_id,
        score=0.9,
    )
    assert info.content == "Test content"
    assert info.summary == "Test summary"
    assert info.metadata == {"test_key": "test_value"}
    assert info.id == info_id
    assert info.score == 0.9


def test_content_property():
    media = MockMedia("Another content")
    info = Information(media=media)
    assert info.content == "Another content"


def test_id_property():
    custom_id = "custom-doc-id"
    info = Information(media=MockMedia("Content"), id=custom_id)
    assert info.id == custom_id


def test_score_property():
    info = Information(media=MockMedia("Score test"))
    assert info.score is None
    info.score = 0.8
    assert info.score == 0.8


def test_add_child():
    parent = Information(media=MockMedia("Parent"))
    child = Information(media=MockMedia("Child"))
    parent.add_child(child)
    assert child in parent.children
    assert child.parent == parent


def test_compute_embedding_no_children():
    info = Information(media=MockMedia("Short text"))
    embedding = info.compute_embedding(embedding_function=simple_embedding_function)
    assert embedding == simple_embedding_function("Short text")
    assert info.embedding == embedding


def test_compute_embedding_with_children():
    parent = Information(media=MockMedia("Parent info"))
    child1 = Information(media=MockMedia("Child 1"))
    child2 = Information(media=MockMedia("Child 2"))
    parent.add_child(child1)
    parent.add_child(child2)
    embedding = parent.compute_embedding(embedding_function=simple_embedding_function)

    expected = np.mean(
        [
            simple_embedding_function("Child 1"),
            simple_embedding_function("Child 2"),
            simple_embedding_function("Parent info"),
        ],
        axis=0,
    ).tolist()

    for actual, exp in zip(embedding, expected):
        assert pytest.approx(actual) == exp


def test_summarize_custom_function():
    info = Information(media=MockMedia("This is a test document with multiple words"))
    summary = info.summarize(summarization_function=simple_summarization_function)
    assert summary == "This..."


@pytest.mark.skip(reason="Requires external API call")
def test_summarize_default_function():
    info = Information(media=MockMedia("Test content for default summarizer"))
    summary = info.summarize()
    assert summary is not None


def test_get_all_informations():
    root = Information(media=MockMedia("Root"))
    child1 = Information(media=MockMedia("Child 1"))
    child2 = Information(media=MockMedia("Child 2"))
    grandchild = Information(media=MockMedia("Grandchild"))
    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(grandchild)
    all_infos = root.get_all_informations()
    assert len(all_infos) == 4
    assert root in all_infos
    assert child1 in all_infos
    assert child2 in all_infos
    assert grandchild in all_infos


def test_copy():
    original = Information(
        media=MockMedia("Copy test content"),
        summary="Test summary",
        metadata={"key": "value"},
        id="copy-id",
        score=0.95,
    )
    cloned = original.__copy__()
    assert cloned is not original
    assert cloned.media.get_content() == original.media.get_content()
    assert cloned.summary == original.summary
    assert cloned.metadata == original.metadata
    assert cloned.id == original.id
    assert cloned.score == original.score
    assert cloned.parent is None
    assert cloned.children == []


def test_contains():
    info = Information(media=MockMedia("Test content"))
    assert "Test" in info
    assert "content" in info
    assert "random" not in info


def test_str():
    info = Information(media=MockMedia("String representation"))
    assert str(info) == "String representation"


def test_hash():
    info1 = Information(media=MockMedia("Same content"))
    info2 = Information(media=MockMedia("Same content"))
    info3 = Information(media=MockMedia("Different content"))
    assert hash(info1) == hash(info2)
    assert hash(info1) != hash(info3)


def test_eq():
    info1 = Information(media=MockMedia("Equal content"))
    info2 = Information(media=MockMedia("Equal content"))
    info3 = Information(media=MockMedia("Not equal"))
    assert info1 == info2
    assert info1 != info3


def test_ne():
    info1 = Information(media=MockMedia("Different check"))
    info2 = Information(media=MockMedia("Different check"))
    info3 = Information(media=MockMedia("Totally different"))
    assert not (info1 != info2)
    assert info1 != info3
