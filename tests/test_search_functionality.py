import pytest
from search_module import SearchEngine  # assuming there's a search engine module

# sample data for testing
test_data = [
    {"id": 1, "content": "artificial intelligence and machine learning"},
    {"id": 2, "content": "deep learning and neural networks"},
    {"id": 3, "content": "natural language processing"},
]

@pytest.fixture
def search_engine():
    # initialize the search engine with test data
    engine = SearchEngine()
    engine.index_documents(test_data)
    return engine

def test_search_single_term(search_engine):
    results = search_engine.search("artificial")
    assert len(results) == 1
    assert results[0]["id"] == 1

def test_search_multiple_terms(search_engine):
    results = search_engine.search("deep learning")
    assert len(results) == 1
    assert results[0]["id"] == 2

def test_search_no_results(search_engine):
    results = search_engine.search("quantum computing")
    assert len(results) == 0

def test_search_partial_match(search_engine):
    results = search_engine.search("machine")
    assert len(results) == 1
    assert results[0]["id"] == 1

def test_search_case_insensitivity(search_engine):
    results = search_engine.search("NATURAL LANGUAGE PROCESSING")
    assert len(results) == 1
    assert results[0]["id"] == 3

# TODO: add more tests for edge cases and performance