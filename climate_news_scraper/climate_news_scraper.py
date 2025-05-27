# climate_news_scraper_simplified.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import traceback

class MaharashtraClimateNewsScraper:
    def __init__(self):
        # Initialize Chrome options
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless=new")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36")
        
        # Simple queries focused on Maharashtra weather events
        self.queries = [
            "Maharashtra flood",
            "Maharashtra drought",
            "Maharashtra monsoon",
            "Maharashtra rainfall",
            "Maharashtra heatwave"
        ]
    
    def setup_driver(self):
        """Initialize and return a Chrome WebDriver"""
        return webdriver.Chrome(options=self.chrome_options)
    
    def scrape_news(self, driver):
        """Scrape news using direct Google search instead of Google News"""
        all_articles = []
        
        for query in self.queries:
            # Use regular Google search instead of Google News
            url = f"https://www.google.com/search?q={query}+news&tbm=nws"
            print(f"Searching for: {query}")
            
            try:
                driver.get(url)
                time.sleep(2)  # Wait for page to load
                
                # Find all news article elements
                article_elements = driver.find_elements(By.CSS_SELECTOR, "div.SoaBEf")
                print(f"Found {len(article_elements)} news articles")
                
                for article in article_elements:
                    try:
                        # Get headline
                        headline_element = article.find_element(By.CSS_SELECTOR, "div.mCBkyc")
                        headline = headline_element.text.strip()
                        
                        # Get link
                        link_element = article.find_element(By.CSS_SELECTOR, "a")
                        link = link_element.get_attribute("href")
                        
                        if headline and link:
                            all_articles.append({
                                "Headline": headline,
                                "Link": link
                            })
                            print(f"Added: {headline[:40]}...")
                    except Exception as e:
                        print(f"Error extracting article details: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error searching for {query}: {str(e)}")
                print(traceback.format_exc())
                continue
        
        return all_articles
    
    def run_scraper(self):
        """Main method to run the scraper"""
        driver = None
        
        try:
            driver = self.setup_driver()
            print("Successfully initialized Chrome driver")
            
            all_articles = self.scrape_news(driver)
            
            # Remove duplicates based on links
            unique_links = set()
            unique_articles = []
            
            for article in all_articles:
                if article["Link"] not in unique_links:
                    unique_links.add(article["Link"])
                    unique_articles.append(article)
            
            # Convert to DataFrame and save as CSV
            if unique_articles:
                df = pd.DataFrame(unique_articles)
                csv_filename = f"maharashtra_climate_news_{time.strftime('%Y%m%d-%H%M%S')}.csv"
                df.to_csv(csv_filename, index=False)
                print(f"\nSaved {len(unique_articles)} unique articles to {csv_filename}")
                
                # Display the dataframe in terminal
                print("\nScraping Results:")
                print(df.head(10))  # Show first 10 results
                if len(df) > 10:
                    print(f"...and {len(df) - 10} more results")
                
                return csv_filename
            else:
                print("\nNo articles were found.")
                return None
        except Exception as e:
            print(f"Error during scraping: {e}")
            print(traceback.format_exc())
            return None
        finally:
            if driver:
                driver.quit()

if __name__ == "__main__":
    print("Starting Maharashtra Climate News Scraper")
    start_time = time.time()
    scraper = MaharashtraClimateNewsScraper()
    scraper.run_scraper()
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")