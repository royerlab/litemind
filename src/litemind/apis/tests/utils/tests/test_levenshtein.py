import pytest

from litemind.apis.tests.utils.levenshtein import levenshtein_distance


def test_levenshtein_distance():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("flaw", "lawn") == 2
    assert levenshtein_distance("intention", "execution") == 5
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("a", "") == 1
    assert levenshtein_distance("", "a") == 1
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("abc", "def") == 3


if __name__ == "__main__":
    pytest.main()
