import pytest
from search_module import SearchEngine  # assuming there's a search engine class 

# test data setup
@pytest.fixture
def search_engine():
    # create a search engine instance for testing
    engine = SearchEngine()
    engine.add_document("First document content")
    engine.add_document("Second document content")
    return engine

def test_search_single_term(search_engine):
    # test searching for a single term
    results = search_engine.search("First")
    assert len(results) == 1
    assert results[0] == "First document content"

def test_search_multiple_terms(search_engine):
    # test searching for multiple terms
    results = search_engine.search("content")
    assert len(results) == 2  # both documents should match
    assert "First document content" in results
    assert "Second document content" in results

def test_search_no_results(search_engine):
    # test searching for a term that doesn't exist
    results = search_engine.search("Nonexistent")
    assert len(results) == 0

def test_search_case_insensitivity(search_engine):
    # test that search is case insensitive
    results = search_engine.search("first")
    assert len(results) == 1
    assert results[0] == "First document content"

# TODO: add more edge cases and input validation tests later