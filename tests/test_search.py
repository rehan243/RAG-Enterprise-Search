import pytest
from search_module import SearchEngine  # assuming this is where the search logic is

@pytest.fixture
def search_engine():
    # set up a search engine instance for testing
    engine = SearchEngine()
    engine.index_data(["apple", "banana", "cherry"])  # indexing some sample data
    return engine

def test_search_existing_term(search_engine):
    # test searching for an existing term
    results = search_engine.search("apple")
    assert results == ["apple"]

def test_search_non_existing_term(search_engine):
    # test searching for a term that doesn't exist
    results = search_engine.search("orange")
    assert results == []

def test_search_partial_term(search_engine):
    # test searching for a term that partially matches
    results = search_engine.search("ban")
    assert results == ["banana"]

def test_search_case_insensitive(search_engine):
    # test that search is case insensitive
    results = search_engine.search("CHERRY")
    assert results == ["cherry"]

def test_index_data(search_engine):
    # test if indexing data works properly
    search_engine.index_data(["date"])
    results = search_engine.search("date")
    assert results == ["date"]

# TODO: add more tests for edge cases