import pytest
from search_module import SearchEngine

# simple test data
mock_data = [
    {"id": 1, "content": "apple pie recipe"},
    {"id": 2, "content": "banana bread recipe"},
    {"id": 3, "content": "cherry tart recipe"},
]

@pytest.fixture
def search_engine():
    # initialize the search engine with mock data
    engine = SearchEngine()
    engine.index_data(mock_data)
    return engine

def test_search_exact_match(search_engine):
    results = search_engine.search("apple pie recipe")
    assert len(results) == 1
    assert results[0]["id"] == 1

def test_search_partial_match(search_engine):
    results = search_engine.search("banana")
    assert len(results) == 1
    assert results[0]["id"] == 2

def test_search_no_results(search_engine):
    results = search_engine.search("pizza")
    assert len(results) == 0

def test_search_case_insensitivity(search_engine):
    results = search_engine.search("CHERRY TART RECIPE")
    assert len(results) == 1
    assert results[0]["id"] == 3

# TODO: add more tests for edge cases and performance