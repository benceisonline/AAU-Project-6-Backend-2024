# pytest tests/test_supabase_utils.py
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Now import modules from project root directory
from api_utilities.supabase_utils import fetch_data_from_supabase, fetch_articles, fetch_newest_articles

# Mock data for testing
mock_supabase_response = {
    "data": [
        {"user_id": 1, "article_id": 101, "published_time": "2023-01-01T00:00:00Z"},
        {"user_id": 2, "article_id": 102, "published_time": "2023-01-02T00:00:00Z"}
    ]
}

@pytest.fixture
def supabase_mock():
    with patch('api_utilities.supabase_utils.supabase') as mock_supabase:
        yield mock_supabase

def test_fetch_data_from_supabase(supabase_mock):
    supabase_mock.table.return_value.select.return_value.order.return_value.execute.return_value = MagicMock(data=mock_supabase_response['data'])
    
    result = fetch_data_from_supabase('LightFM')
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert list(result.columns) == ['user_id', 'article_id', 'published_time']

def test_fetch_data_from_supabase_no_data(supabase_mock):
    supabase_mock.table.return_value.select.return_value.order.return_value.execute.return_value = MagicMock(data=None)
    
    with pytest.raises(ValueError, match="Failed to fetch data from table LightFM"):
        fetch_data_from_supabase('LightFM')

def test_fetch_articles(supabase_mock):
    article_ids = [101, 102]
    supabase_mock.table.return_value.select.return_value.in_.return_value.execute.return_value = MagicMock(data=mock_supabase_response['data'])
    
    result = fetch_articles(article_ids)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]['article_id'] == 101

def test_fetch_articles_no_data(supabase_mock):
    article_ids = [101, 102]
    supabase_mock.table.return_value.select.return_value.in_.return_value.execute.return_value = MagicMock(data=None)
    
    with pytest.raises(ValueError, match="No articles found with the given IDs"):
        fetch_articles(article_ids)

def test_fetch_newest_articles(supabase_mock):
    start_index = 0
    no_recommendations = 2
    supabase_mock.table.return_value.select.return_value.order.return_value.range.return_value.execute.return_value = MagicMock(data=mock_supabase_response['data'])
    
    result = fetch_newest_articles(start_index, no_recommendations)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]['article_id'] == 101

def test_fetch_newest_articles_no_data(supabase_mock):
    start_index = 0
    no_recommendations = 2
    supabase_mock.table.return_value.select.return_value.order.return_value.range.return_value.execute.return_value = MagicMock(data=None)
    
    with pytest.raises(ValueError, match="No articles found with the given range"):
        fetch_newest_articles(start_index, no_recommendations)
