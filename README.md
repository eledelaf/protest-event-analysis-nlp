The code from the project is separated by folders. 
### 1.MediaCloud:
    * URLS.numbers: Is the csv that I got from MediaCLoud

### 2.Data_cleaning
    * FirstQuery.py: Is the python code used to select the papers that are going to be examined later. Also clean the URLs files

### 3.web_scrapping
    * fun_scrap3.py:
    * scrape_to_mongo.py:
    * URLs_clean.csv:

### 4.class_hf
    * threshold.py: I used this code to find the optimal threshold
    * hf_class.py: Classification function 
    * run_hf.py: Runs the function from hf_class.py throught the MongoDB data base
    * hf_results.py: This script analyzes the results of the Hugging Face classifier.
    
### 5.sentiment
    * sent_analysis.py: The code to do the sentiment analysis


### 6.Topic_analysis
    * topic_modeling.py
    * topic_modeling:
        * artices_with_topics.csv
        * representative_docs.csv
        * topic_info.csv 
        * topic_share_by_paper_time.csv
        * topic_share_by_paper.csv
        * topic_share_by_time.csv
        * topics_barchart.html
        * topic_map.html

### 7.1plots
    * check_art_title.py
    * check_sent_newspapers.py
    * check_sentiment.py
    * plot_sent.py
    * plot_sentiment_by_paper.py
    * plot_sentiment_composition_over_time.py
    * plot_sentiment_density_by_period.py
    * plot_sentiment.py
    * plot_topic_by_time.py
    * protest_by_year.py
    * protest_covid_period.py
    * vader_covid.py
    * weekly_protest_peaks.py

### 7.2figures
    * sentiment_composition_over_time.png
    * sentiment_density_by_period.png
    * sentiment_distribution_by_paper_by_period.png
    * sentiment_distribution_by_paper.png
    * sentiment_distribution_by_period.png
    * sentiment_distribution_overall.png
    * sentiment_heatmap_paper_time.png
    * sentiment_monthly_mean_facets.png
    * sentiment_monthly_mean_overall.png
    * sentiment_over_time.png
    * sentiment_weekly_mean_overall.png
    * topic_by_time.png

### 7.3outputs
    * weekly_protest_counts_by_paper.csv
    * weekly_protest_counts.csv
    * protest_counts_by_covid_period_and_outlet.csv
    * protest_counts_long.csv
    * protest_counts_by_year_outlet.csv
    * sentiment_mean_pivot_outlet_x_covid_period.csv
    * sentiment_mean_by_outlet_and_covid_period.csv
    * sentiment_mean_overall_by_covid_period.csv
    
    
    
    
    

        
