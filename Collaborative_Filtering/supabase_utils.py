import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv('../.env')

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_ANON_KEY")

supabase: Client = create_client(url, key)

def fetch_data_from_supabase(table_name: str):
    response = supabase.table(table_name).select("*").order('user_id', desc=True).execute()
    if response.data is None:
        raise ValueError(f"Failed to fetch data from table {table_name}")
    return pd.DataFrame(response.data)

def fetch_articles(article_ids: list):
    response = supabase.table('Articles').select('*').in_('article_id', article_ids).execute()
    if response.data is None:
        raise ValueError("No articles found with the given IDs")
    return response.data

def fetch_newest_articles(start_index: int, no_recommendations: int):
    response = supabase.table('Articles').select('*').order('published_time', desc=True).range(start_index, start_index + no_recommendations - 1).execute()
    if response.data is None:
        raise ValueError("No articles found with the given range")
    return response.data
