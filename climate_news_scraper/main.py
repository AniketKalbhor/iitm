# main.py
import subprocess # Launching the RSS Scraper
import sys
import time

def run_scraper():
    """Run the climate news scraper"""
    print("=== MAHARASHTRA CLIMATE NEWS SCRAPER ===")
    print("Fetching climate news articles from the past 6 months")
    
    start_time = time.time()
    
    # Run the RSS scraper and capture output
    result = subprocess.run([sys.executable, "maharashtra_climate_news_rss.py"], 
                           capture_output=True, text=True)
    
    # Print the output from the scraper
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    # Extract CSV filename from output
    csv_filename = None
    for line in result.stdout.split('\n'):
        if line.startswith("Saved") and line.endswith(".csv"):
            csv_filename = line.split()[-1]
    
    if csv_filename:
        # Run the analyzer on the CSV file
        print("\n=== ANALYZING RESULTS ===")
        analyze_result = subprocess.run([sys.executable, "climate_news_analyzer.py", csv_filename],
                                       capture_output=True, text=True)
        print(analyze_result.stdout)
        if analyze_result.stderr:
            print("Analysis Errors:")
            print(analyze_result.stderr)
    
    elapsed_time = time.time() - start_time
    print(f"\nProcess complete! Total time: {elapsed_time:.2f} seconds")
    if csv_filename:
        print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    run_scraper()