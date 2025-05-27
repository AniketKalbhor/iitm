# climate_news_analyzer.py
import pandas as pd
import sys
import time

class ClimateNewsAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        try:
            self.data = pd.read_csv(csv_file)
            print(f"Successfully loaded {len(self.data)} articles from {csv_file}")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            self.data = pd.DataFrame(columns=["headline", "url", "sentiment", "keyword"])
        
        # Impact categories for analysis (simple version)
        self.impact_categories = {
            "drought": "water scarcity",
            "rainfall": "water resources", 
            "flood": "disaster impact",
            "heatwave": "health effects",
            "monsoon": "agricultural impact"
        }
    
    def analyze_articles(self):
        """Analyze each article for climate impact categories"""
        if len(self.data) == 0:
            print("No articles to analyze")
            return self.data
            
        # Add new column for impact category based on keyword
        self.data["impact_category"] = self.data["keyword"].map(
            lambda k: self.impact_categories.get(k, "general impact")
        )
        
        output_file = f"analyzed_{self.csv_file}"
        self.data.to_csv(output_file, index=False)
        
        print(f"\nAnalysis complete. Results saved to {output_file}")
        return self.data
    
    def generate_summary(self):
        """Generate summary statistics"""
        if len(self.data) == 0:
            print("No data to summarize")
            return
            
        # Basic statistics
        total_articles = len(self.data)
        keyword_counts = self.data["keyword"].value_counts()
        
        # Extract sentiment labels without scores
        self.data["sentiment_label"] = self.data["sentiment"].str.split("(").str[0].str.strip()
        sentiment_counts = self.data["sentiment_label"].value_counts()
        
        print("\n===== MAHARASHTRA CLIMATE NEWS SUMMARY =====")
        print(f"Total articles analyzed: {total_articles}")
        
        print("\nKeyword Distribution:")
        for keyword, count in keyword_counts.items():
            print(f"  - {keyword}: {count} articles")
            
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"  - {sentiment}: {count} articles")
        
        if "impact_category" in self.data.columns:
            impact_counts = self.data["impact_category"].value_counts()
            print("\nImpact Categories:")
            for impact, count in impact_counts.items():
                print(f"  - {impact}: {count} articles")
        
        print("\nTop Headlines:")
        for i, row in self.data.head(min(5, len(self.data))).iterrows():
            print(f"  - {row['headline']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python climate_news_analyzer.py <csv_file>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    start_time = time.time()
    analyzer = ClimateNewsAnalyzer(csv_file)
    analyzer.analyze_articles()
    analyzer.generate_summary()
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")