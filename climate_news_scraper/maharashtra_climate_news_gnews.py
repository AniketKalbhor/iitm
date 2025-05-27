# maharashtra_climate_news_gnews.py
import requests
import pandas as pd
import time

class MaharashtraClimateNewsGNews:
    def __init__(self):
        # Climate and weather keywords
        self.keywords = [
            "Maharashtra flood",
            "Maharashtra drought",
            "Maharashtra rainfall",
            "Maharashtra heatwave",
            "Maharashtra monsoon"
        ]
        
        # Base URL for GNews API
        self.base_url = "https://gnews.io/api/v4/search"
        
    def fetch_articles(self, query):
        """Fetch articles for a specific query"""
        params = {
            'q': query,
            'lang': 'en',
            'country': 'in',
            'max': 10  # Number of results per query
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            
            if 'articles' in data:
                return data['articles']
            else:
                print(f"API Error: {data.get('errors', ['Unknown error'])}")
                return []
        except Exception as e:
            print(f"Error fetching articles for query '{query}': {e}")
            return []
    
    def run_api_search(self):
        """Main method to run the API search"""
        all_articles = []
        
        # Search for each keyword
        for keyword in self.keywords:
            print(f"Searching for: {keyword}")
            articles = self.fetch_articles(keyword)
            print(f"Found {len(articles)} articles for '{keyword}'")
            all_articles.extend(articles)
            # Be nice to the API with a short delay
            time.sleep(1)
        
        # Remove duplicates based on URL
        unique_urls = set()
        unique_articles = []
        
        for article in all_articles:
            if article['url'] not in unique_urls:
                unique_urls.add(article['url'])
                # Extract only needed fields
                unique_articles.append({
                    'Headline': article['title'],
                    'Link': article['url'],
                    'Published': article['publishedAt']
                })
        
        # Convert to DataFrame and save as CSV
        if unique_articles:
            df = pd.DataFrame(unique_articles)
            csv_filename = f"maharashtra_climate_news_{time.strftime('%Y%m%d-%H%M%S')}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"\nSaved {len(unique_articles)} unique articles to {csv_filename}")
            
            # Display the dataframe in terminal
            print("\nSearch Results:")
            print(df[['Headline', 'Link']].head(10))  # Show first 10 results
            if len(df) > 10:
                print(f"...and {len(df) - 10} more results")
            
            return csv_filename
        else:
            print("\nNo articles were found.")
            return None

if __name__ == "__main__":
    print("Starting Maharashtra Climate News GNews Search")
    start_time = time.time()
    gnews_api = MaharashtraClimateNewsGNews()
    gnews_api.run_api_search()
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")