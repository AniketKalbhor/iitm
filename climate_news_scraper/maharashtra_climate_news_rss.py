# maharashtra_climate_news_rss.py
import feedparser
import pandas as pd
import time
import re
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

class MaharashtraClimateNewsRSS:
    def __init__(self):
        # Climate keywords with weights - English only
        self.climate_keywords = {
            "drought": 3, "rainfall": 3, "flood": 3, "heatwave": 3, "monsoon": 3,
            "climate change": 5, "global warming": 4, "heavy rain": 3, "water crisis": 4, 
            "temperature rise": 3, "extreme weather": 4, "water scarcity": 4,
            "climate": 2, "rain": 1, "weather": 1, "temperature": 1,
            "crop damage": 3, "agriculture": 2, "irrigation": 2
        }
        
        # Maharashtra keywords with weights - English only
        self.location_keywords = {
            "maharashtra": 5, "mumbai": 4, "pune": 3, "nagpur": 3, "nashik": 3,
            "aurangabad": 3, "solapur": 3, "kolhapur": 3, "thane": 3, "konkan": 3,
            "vidarbha": 3, "marathwada": 3, "western maharashtra": 4
        }
        
        # RSS feeds from English-language news sources
        self.rss_feeds = [
            "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
            "https://www.indiatoday.in/rss/home",
            "https://www.hindustantimes.com/feeds/rss/india/rssfeed.xml",
            "https://indianexpress.com/feed/",
            "https://www.ndtv.com/rss",
            "https://www.livemint.com/rss/news",
            "https://www.thehindu.com/news/national/feeder/default.rss",
            # Maharashtra specific feeds
            "https://timesofindia.indiatimes.com/rssfeeds/-2128838597.cms",  # Maharashtra TOI
            "https://www.hindustantimes.com/feeds/rss/cities/mumbai-news/rssfeed.xml",  # HT Mumbai
            "https://indianexpress.com/section/cities/mumbai/feed/",  # IE Mumbai
            "https://indianexpress.com/section/cities/pune/feed/"  # IE Pune
        ]
    
    def get_article_content(self, url):
        """Fetch and extract content from the article URL"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            return ""
    
    def calculate_relevance_score(self, text, keyword_dict):
        """Calculate a relevance score based on weighted keywords"""
        text_lower = text.lower()
        score = 0
        
        for keyword, weight in keyword_dict.items():
            # Count exact matches
            count = text_lower.count(keyword.lower())
            score += count * weight
            
        return score
    
    def is_recent(self, entry, max_months=6):
        """Check if article is from the last 6 months"""
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
                now = datetime.now()
                return (now - pub_date) <= timedelta(days=max_months*30)
            
            # Default: include if we can't determine recency
            return True
        except Exception:
            # Default: include if date processing fails
            return True
    
    def extract_date(self, entry):
        """Extract publication date from entry"""
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                return datetime(*entry.published_parsed[:6]).strftime('%Y-%m-%d')
            
            if hasattr(entry, 'published'):
                try:
                    # Try basic format parsing - this will work for many RSS feeds
                    return entry.published.split('T')[0]  # Often ISO format like 2023-05-08T14:30:00
                except:
                    return entry.published[:10]  # Try to extract just the date portion
            
            return "Unknown"
        except Exception:
            return "Unknown"
    
    def is_english(self, text):
        """Simple check if text is in English - checking for common English words"""
        english_markers = ["the", "and", "in", "of", "to", "a", "for", "with", "that", "on"]
        text_lower = text.lower()
        # Count how many English marker words are in the text
        marker_count = sum(1 for marker in english_markers if f" {marker} " in f" {text_lower} ")
        # If at least 3 common English words are present, consider it English
        return marker_count >= 3
    
    def fetch_and_filter_articles(self, min_relevance_score=5):
        """Fetch articles from RSS feeds and filter for climate news in Maharashtra with improved relevance"""
        all_articles = []
        
        for feed_url in self.rss_feeds:
            try:
                print(f"Fetching from: {feed_url}")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    # Skip if not recent (last 6 months)
                    if not self.is_recent(entry, max_months=6):
                        continue
                        
                    title = entry.title if hasattr(entry, 'title') else ""
                    summary = entry.summary if hasattr(entry, 'summary') else ""
                    
                    # Check if content appears to be in English
                    if not self.is_english(f"{title} {summary}"):
                        continue
                    
                    # Combine title and summary for initial screening
                    initial_text = f"{title} {summary}".lower()
                    
                    # Initial screening for at least one climate keyword and one location keyword
                    has_climate_keyword = any(keyword.lower() in initial_text for keyword in self.climate_keywords)
                    has_location_keyword = any(keyword.lower() in initial_text for keyword in self.location_keywords)
                    
                    # Only proceed with full content analysis if initial screening passes
                    if has_climate_keyword and has_location_keyword:
                        # If needed, get full content for better analysis
                        print(f"Found potential match: {title}")
                        try:
                            full_content = self.get_article_content(entry.link)
                            all_content = f"{title} {summary} {full_content}"
                        except:
                            # If content fetch fails, just use title and summary
                            all_content = f"{title} {summary}"
                        
                        # Calculate separate scores
                        climate_score = self.calculate_relevance_score(all_content, self.climate_keywords)
                        location_score = self.calculate_relevance_score(all_content, self.location_keywords)
                        
                        # Combined relevance score - we want both climate and location to be relevant
                        # Taking the minimum ensures both aspects must be present
                        relevance_score = min(climate_score, location_score/2)
                        
                        # Only include if relevance score is above threshold
                        if relevance_score >= min_relevance_score:
                            # Extract date
                            pub_date = self.extract_date(entry)
                            
                            # Extract primary keyword - which climate term is most frequent
                            primary_keyword = max(self.climate_keywords.items(), 
                                                 key=lambda kw: all_content.lower().count(kw[0].lower()))[0]
                            
                            # Create article entry
                            article = {
                                'headline': entry.title,
                                'date': pub_date,
                                'url': entry.link,
                                'keyword': primary_keyword,
                                'sentiment': "Neutral (0.0)",  # Default sentiment
                                'relevance_score': relevance_score
                            }
                            all_articles.append(article)
                            print(f"Found relevant article: {entry.title} (Score: {relevance_score})")
            
            except Exception as e:
                print(f"Error fetching from {feed_url}: {e}")
                continue
        
        return all_articles
    
    def run_rss_search(self):
        """Main method to run the RSS search"""
        print("Fetching climate news about Maharashtra from RSS feeds...")
        start_time = time.time()
        all_articles = self.fetch_and_filter_articles()
        fetch_time = time.time() - start_time
        print(f"Fetching completed in {fetch_time:.2f} seconds, found {len(all_articles)} articles")
        
        # Remove duplicates based on headlines (case-insensitive)
        unique_headlines = set()
        unique_articles = []
        
        for article in all_articles:
            headline_lower = article['headline'].lower()
            if headline_lower not in unique_headlines:
                unique_headlines.add(headline_lower)
                unique_articles.append(article)
        
        # Sort by relevance score
        sorted_articles = sorted(unique_articles, key=lambda x: x['relevance_score'], reverse=True)
        
        # Convert to DataFrame and save as CSV
        if sorted_articles:
            df = pd.DataFrame(sorted_articles)
            csv_filename = f"maharashtra_climate_news_{time.strftime('%Y%m%d-%H%M%S')}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"\nSaved {len(sorted_articles)} unique articles to {csv_filename}")
            
            # Display the top 10 results in terminal
            print("\nTop Results:")
            if len(sorted_articles) > 10:
                display_articles = sorted_articles[:10]
            else:
                display_articles = sorted_articles
                
            for idx, article in enumerate(display_articles, 1):
                print(f"{idx}. {article['headline']} (Score: {article['relevance_score']:.1f})")
                print(f"   Date: {article['date']} | Keyword: {article['keyword']}")
                print(f"   URL: {article['url']}")
                print()
            
            return csv_filename
        else:
            print("\nNo relevant articles were found.")
            return None

if __name__ == "__main__":
    print("Starting Maharashtra Climate News RSS Search")
    start_time = time.time()
    rss_feed = MaharashtraClimateNewsRSS()
    rss_feed.run_rss_search()
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")