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
    article_type: str
    url: str
    category: int
    category_str: str
    sentiment_score: float
    sentiment_label: str
    image_ids: Optional[list[int]]
    image_url: str
