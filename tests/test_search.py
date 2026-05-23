import pytest
from search_module import SearchEngine  # assuming we have a SearchEngine class

@pytest.fixture
def search_engine():
    # setup a search engine instance before each test
    engine = SearchEngine()
    yield engine
    # teardown if necessary

def test_search_returns_results(search_engine):
    # test that search returns results for a valid query
    query = "enterprise RAG"
    results = search_engine.search(query)
    assert len(results) > 0  # we expect some results

def test_search_empty_query(search_engine):
    # test that searching with an empty query returns an empty result set
    query = ""
    results = search_engine.search(query)
    assert len(results) == 0  # no results expected

def test_search_case_insensitivity(search_engine):
    # test that the search is case insensitive
    query = "Enterprise rag"
    results_upper = search_engine.search(query)
    results_lower = search_engine.search(query.lower())
    assert results_upper == results_lower  # results should be the same

def test_search_special_characters(search_engine):
    # test that special characters don't break the search
    query = "RAG@2023!"
    results = search_engine.search(query)
    assert len(results) >= 0  # should handle special chars without error

# TODO: add more tests for edge cases and performance if needed