import pytest
from search_module import SearchEngine  # assuming this is the module for search functionality

# setup a fixture for the search engine
@pytest.fixture
def search_engine():
    engine = SearchEngine()
    # TODO: maybe seed some data here if needed
    return engine

def test_search_returns_results(search_engine):
    query = "example search term"
    results = search_engine.search(query)
    # check that results are not empty
    assert results, "search should return results for a valid query"

def test_search_returns_correct_type(search_engine):
    query = "another term"
    results = search_engine.search(query)
    # check that results are of expected type
    assert isinstance(results, list), "search results should be a list"

def test_search_empty_query(search_engine):
    query = ""
    results = search_engine.search(query)
    # check that empty query returns no results
    assert not results, "search with an empty query should return no results"

def test_search_no_results(search_engine):
    query = "nonexistent term"
    results = search_engine.search(query)
    # check that a query with no matches returns an empty list
    assert results == [], "search for a term that doesn't exist should return an empty list"

# TODO: add more tests for edge cases and different query types