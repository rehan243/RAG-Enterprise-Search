import pytest
from search_module import SearchEngine  # assuming this is where the search logic is

# assuming we have some predefined data for testing
test_data = [
    {"id": 1, "content": "the quick brown fox"},
    {"id": 2, "content": "jumps over the lazy dog"},
    {"id": 3, "content": "hello world"},
]

@pytest.fixture
def search_engine():
    # setup a search engine instance
    engine = SearchEngine()
    engine.index_documents(test_data)  # indexing test data
    return engine

def test_search_by_keyword(search_engine):
    # search for a keyword
    results = search_engine.search("quick")
    assert len(results) == 1
    assert results[0]["id"] == 1

def test_search_no_results(search_engine):
    # search for a term that doesn't exist
    results = search_engine.search("nonexistent")
    assert len(results) == 0

def test_search_multiple_keywords(search_engine):
    # search for multiple keywords
    results = search_engine.search("lazy dog")
    assert len(results) == 1
    assert results[0]["id"] == 2

def test_search_partial_match(search_engine):
    # check if partial match works
    results = search_engine.search("hello")
    assert len(results) == 1
    assert results[0]["id"] == 3

# TODO: add more tests for edge cases and error handling