import pytest
from search_module import SearchEngine  # assuming the search engine is in search_module

# test the search functionality of the SearchEngine class
def test_search_with_valid_query():
    search_engine = SearchEngine()
    query = "machine learning"
    results = search_engine.search(query)
    
    # check if results is a list
    assert isinstance(results, list)
    # check if we get some results back
    assert len(results) > 0

def test_search_with_empty_query():
    search_engine = SearchEngine()
    query = ""
    results = search_engine.search(query)
    
    # expect no results for an empty query
    assert results == []

def test_search_with_non_existent_query():
    search_engine = SearchEngine()
    query = "this query does not exist"
    results = search_engine.search(query)
    
    # expect results to be empty
    assert results == []

# TODO: add more tests to check result contents and ranking if needed
def test_search_case_insensitivity():
    search_engine = SearchEngine()
    query = "Machine Learning"
    results_upper = search_engine.search(query)
    results_lower = search_engine.search(query.lower())
    
    # expect the same results regardless of case
    assert results_upper == results_lower