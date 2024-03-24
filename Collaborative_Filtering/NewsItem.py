from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class NewsItem(BaseModel):
    article_id: int
    title: str
    subtitle: str
    last_modified_time: datetime
    premium: bool
    body: str
    published_time: datetime
    image_ids: Optional[list[int]]
    article_type: str
    url: str
    entity_groups: list[str]
    topics: list[str]
    category: int
    subcategory: list[int]
    category_str: str
    total_inviews: Optional[float]
    total_pageviews: Optional[float]
    total_read_time: Optional[float]
    sentiment_score: float
    sentiment_label: str