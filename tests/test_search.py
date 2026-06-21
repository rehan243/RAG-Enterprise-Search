import pytest
from search_module import SearchEngine  # assuming there's a search_module

@pytest.fixture
def search_engine():
    # setup a search engine instance before each test
    engine = SearchEngine()
    engine.index_documents([
        {"id": 1, "content": "hello world"},
        {"id": 2, "content": "goodbye world"},
    ])
    return engine

def test_search_existing_document(search_engine):
    # test searching for existing document
    results = search_engine.search("hello")
    assert len(results) == 1
    assert results[0]['id'] == 1

def test_search_non_existing_document(search_engine):
    # test searching for a non-existing document
    results = search_engine.search("not in index")
    assert len(results) == 0

def test_search_partial_match(search_engine):
    # test searching for a partial match
    results = search_engine.search("world")
    assert len(results) == 2  # both documents should match

def test_search_case_insensitivity(search_engine):
    # test that search is case insensitive
    results = search_engine.search("Hello")
    assert len(results) == 1
    assert results[0]['id'] == 1

# TODO: add more tests for edge cases