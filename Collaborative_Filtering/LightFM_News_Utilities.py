import hashlib
import pandas as pd
import numpy as np

class NewsTools:
    def __init__(self, data):
        self.data = data

    def generate_image_url(self, image_id, article_id, published_timestamp, modified_timestamp):
        hash_string = f"caravaggio-{article_id}-{published_timestamp}-{modified_timestamp}"
        md5_hash = hashlib.md5(hash_string.encode()).hexdigest()
        return f"https://img-cdn-p.ekstrabladet.dk/image/ekstrabladet/{image_id}/relationBig_910/?at={md5_hash}"

    def generate_image_urls(self, recommended_items):
        for item in recommended_items:
            article_id = item['article_id']
            image_ids = item.get('image_ids')
            if image_ids is not None and len(image_ids) > 0:  # Check if image_ids is not None and is not empty
                image_id = image_ids[0]  # Assuming only the first image is used
                published_timestamp = item['published_time'].value / 1000000  # convert to seconds
                modified_timestamp = item['last_modified_time'].value / 1000000  # convert to seconds
                item['image_url'] = self.generate_image_url(image_id, article_id, published_timestamp, modified_timestamp)
            else:
                # If no image IDs are available, use default image URL
                item['image_url'] = "https://is1-ssl.mzstatic.com/image/thumb/Purple112/v4/66/53/28/66532893-b30e-3023-20be-448c4d19428f/AppIcon-0-0-1x_U007ephone-0-85-220.png/1200x630wa.png"

            # Convert NumPy array to list
            if isinstance(item['image_ids'], np.ndarray):
                item['image_ids'] = item['image_ids'].tolist()

        return recommended_items
    
    def get_newest_news(self, news_data):
        column_names = ["article_id", "title", "subtitle", "last_modified_time", "premium", "body", "published_time", 
                        "article_type", "url", "category", "category_str", "sentiment_score", "sentiment_label", "image_ids"]
            
        sorted_news = news_data[column_names].sort_values(by="published_time", ascending=False).head(10)

        news_list = sorted_news.to_dict(orient='records')

        self.generate_image_urls(news_list)

        return {"news": news_list}