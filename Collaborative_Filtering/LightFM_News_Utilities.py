import hashlib
import pandas as pd
import numpy as np
from supabase_utils import supabase

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
                published_timestamp = pd.Timestamp(item['published_time']).value / 1000000
                modified_timestamp = pd.Timestamp(item['last_modified_time']).value / 1000000
                item['image_url'] = self.generate_image_url(image_id, article_id, published_timestamp, modified_timestamp)
            else:
                # If no image IDs are available, use default image URL
                item['image_url'] = "https://is1-ssl.mzstatic.com/image/thumb/Purple112/v4/66/53/28/66532893-b30e-3023-20be-448c4d19428f/AppIcon-0-0-1x_U007ephone-0-85-220.png/1200x630wa.png"

            # Convert NumPy array to list
            if isinstance(item['image_ids'], np.ndarray):
                item['image_ids'] = item['image_ids'].tolist()

        return recommended_items
    
    def get_newest_news(self, request):
        start_index = request.start_index
        no_recommendations = request.no_recommendations

        response = supabase.table('Articles').select('*').order('published_time', desc=True).range(start_index, (start_index + no_recommendations) - 1).execute()

        if response == None:
            raise ValueError("No articles found with the given range")

        self.generate_image_urls(response.data)
        
        return {"news": response.data}

        