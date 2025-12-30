# Organisation of the code
The code from the project is separated by folders. 
* human_sample.csv: The csv were I did the classification

### 1.MediaCloud:
    * URLS.numbers: Is the csv that I got from MediaCLoud

### 2.Data_cleaning
    * FirstQuery.py: Is the python code used to select the papers that are going to be examined later. Also clean the URLs files

### 3.web_scrapping
    * fun_scrap3.py: Function used for scraping
    * scrape_to_mongo.py: gets the urls from URLs_clean.csv, scrapes using fun_scrap3.py and uploads to MongoDB.
    * URLs_clean.csv: Result of FirstQuery.py

### 4.class_hf
    * threshold.py: I used this code to find the optimal threshold
    * hf_class.py: Classification function 
    * run_hf.py: Runs the function from hf_class.py throught the MongoDB data base
    * hf_results.py: This script analyzes the results of the classifier used.
    
### 5.sentiment
    * sent_analysis.py: The code to do the sentiment analysis, and uploads the data to MongoDB

### 6.Topic_analysis
    * topic_modeling.py: Code that does the topic modeling and creates diferent csv files to understand the diferent topics.
    * topic_modeling:
        * artices_with_topics.csv: Csv with the articles and their topics
        * representative_docs.csv: Csv with the 3 most representative articles for each topic
        * topic_info.csv: The information for each topic
        * topic_share_by_paper_time.csv: For each 3 months and for each paper it gives the percentage of apperances by topic
        * topic_share_by_paper.csv: For each news paper gives the percentage of apperance by topic 
        * topic_share_by_time.csv: For every 3 months gives the percentages of apperances by topic
        * topics_barchart.html: For each topic shows the most important words.
        * topic_map.html: Topic map 

### 7.1plots
    * check_sent_newspapers.py: Check sentiment distribution across newspapers. 
    * plot_sentiment_density_by_period.py: Pre vs during vs post COVID distribution of sentiment using VADER compound.Creates density curves for each period, to see if the distribution shifts.
    * plot_topic_by_time.py: This script plots the evolution of topics over time.
    * protest_by_year.py: Determine protest article counts by year and outlet.
    * protest_covid_period.py: Counts PROTEST articles by COVID period (pre/during/post), overall and by outlet.
    * vader_covid.py: Analyze VADER sentiment compound scores by COVID period.
    * weekly_protest_peaks.py: Compute weekly counts of PROTEST-labelled articles from a MongoDB collection and report the weeks with the highest number of PROTEST articles.

### 7.2figures
    * sentiment_density_by_period.png: This comes from plot_sentiment_density_by_period.py
    * topic_by_time.png: This comoes from plot_topic_by_time.py

### 7.3outputs
    * weekly_protest_counts_by_paper.csv: comes from weekly_protest_peaks.py
    * weekly_protest_counts.csv: comes from weekly_protest_peaks.py
    * protest_counts_by_covid_period_and_outlet.csv: from protest_by_year.py
    * protest_counts_long.csv: comes from protest_by_year.py
    * protest_counts_by_year_outlet.csv: comes from protest_by_year.py
    * sentiment_mean_pivot_outlet_x_covid_period.csv: vader_covid.py
    * sentiment_mean_by_outlet_and_covid_period.csv: vader_covid.py
    * sentiment_mean_overall_by_covid_period.csv: vader_covid.py


# How to run the code

##  MongoDB access

 
    
    

        
