import pytest
from search_module import SearchEngine  # assuming this is where the search logic is

@pytest.fixture
def search_engine():
    # create a search engine instance for testing
    engine = SearchEngine()
    return engine

def test_search_returns_results(search_engine):
    # test if search returns results for a valid query
    query = "machine learning"
    results = search_engine.search(query)
    
    # assert we get some results back
    assert len(results) > 0
    assert all(isinstance(result, dict) for result in results)

def test_search_empty_query(search_engine):
    # test search with an empty query returns no results
    query = ""
    results = search_engine.search(query)
    
    # assert we get no results back
    assert len(results) == 0

def test_search_case_insensitivity(search_engine):
    # test that search is case insensitive
    query = "Data Science"
    results_upper = search_engine.search(query.upper())
    results_lower = search_engine.search(query.lower())
    
    # assert both return the same results
    assert results_upper == results_lower

# TODO: add more tests for edge cases and error handling