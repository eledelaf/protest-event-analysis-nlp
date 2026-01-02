# Applying NLP to Protest Event Analysis (UK Media, pre/during/post COVID)

## Overview
This project automates key steps of Protest Event Analysis (PEA) using UK news coverage before and after COVID-19.
It collects candidate protest-related articles from MediaCloud, scrapes full text, stores data in MongoDB, applies a zero-shot protest/not-protest classifier, runs VADER sentiment analysis, and performs topic modelling to study issue patterns over time and across outlets.

**Outlets:** The Guardian, Daily Mail, Evening Standard  
**Period:** 2020-01-01 to 2024-12-31

## Organisation of the code
The project code is organised into folders that follow the pipeline stages.
* human_sample.csv: CSV containing the manually labelled sample used for evaluation.

### 1.MediaCloud:
    * URLS.numbers: Raw CSV export downloaded from MediaCloud.

### 2.Data_cleaning
    * FirstQuery.py: Filters the MediaCloud export to keep only the target outlets and date range, and performs basic URL cleaning. Output: URLs_clean.csv (used as input for scraping).

### 3.web_scrapping
    * fun_scrap3.py: Scraping helper functions.
    * scrape_to_mongo.py: Reads URLs from URLs_clean.csv, scrapes article text using fun_scrap3.py, and uploads results to MongoDB.
    * URLs_clean.csv: Output of FirstQuery.py.

### 4.class_hf
    * threshold.py: Finds/validates the classification threshold (used to convert model confidence into PROTEST / NOT_PROTEST).
    * hf_class.py: Classification functions (Hugging Face zero-shot classification).
    * run_hf.py: Runs the classifier over the MongoDB collection and writes labels scores back to MongoDB.
    * hf_results.py: Analyses classifier outputs (counts, distributions, performance summaries, etc.).
    
### 5.sentiment
    * sent_analysis.py: Runs sentiment analysis (VADER) and uploads sentiment fields back to MongoDB.

### 6.Topic_analysis
    * topic_modeling.py: Runs topic modelling and produces CSV/HTML outputs to inspect topics and trends.
    *topic_modeling/
        artices_with_topics.csv: Articles with assigned topic IDs.
        representative_docs.csv: The 3 most representative documents per topic.
        topic_info.csv: Topic metadata (topic ID, top words, sizes, etc.).
        topic_share_by_paper_time.csv: Topic shares by outlet and quarter.
        topic_share_by_paper.csv: Topic shares by outlet.
        topic_share_by_time.csv: Topic shares over time  in 3-month bins.
        topics_barchart.html: Top words per topic.
        topic_map.html: Topic map visualisation.

### 7.1plots
    * check_sent_newspapers.py: Checks sentiment distributions across newspapers.
    * plot_sentiment_density_by_period.py: Sentiment density curves by COVID period (pre/during/post) using VADER compound.
    * plot_topic_by_time.py: Plots topic prevalence over time.
    * protest_by_year.py: Counts PROTEST-labelled articles by year and outlet.
    * protest_covid_period.py: Counts PROTEST articles by COVID period overall and by outlet.
    * vader_covid.py: Summarises VADER compound scores by COVID period.
    * weekly_protest_peaks.py: Computes weekly PROTEST article counts and reports peak weeks.

### 7.2figures
    * sentiment_density_by_period.png: Output figure from plot_sentiment_density_by_period.py.
    * topic_by_time.png: Output figure from plot_topic_by_time.py.

### 7.3outputs
    * weekly_protest_counts_by_paper.csv: Output from weekly_protest_peaks.py.
    * weekly_protest_counts.csv: Output from weekly_protest_peaks.py.
    * protest_counts_by_covid_period_and_outlet.csv: Output from protest_by_year.py.
    * protest_counts_long.csv: Output from protest_by_year.py.
    * protest_counts_by_year_outlet.csv: Output from protest_by_year.py.
    * sentiment_mean_pivot_outlet_x_covid_period.csv: Output from vader_covid.py.
    * sentiment_mean_by_outlet_and_covid_period.csv: Output from vader_covid.py.
    * sentiment_mean_overall_by_covid_period.csv: Output from vader_covid.py.


# How to run the code

##  MongoDB access
In order to check the data set, the examiner has access to the MongoDB account I created for the project.
user = eledelaf@ucm.es
pasword = Dukvyn-kynmi5-hurmed

If this does not work there is a copy of the two data bases in 8.MongoDB, with a code that downloads the data directly from MongoDB into two .jsonl files. 

## Order of the code:
1. Filer MediaCloud URLs: python 2.Data_cleaning/FirstQuery.py
2. Scrape articles and upload to MongoDB: python 3.web_scrapping/scrape_to_mongo.py
3. Run protest / not-protest classificatio: python 4.class_hf/run_hf.py
4. Run sentiment analysis: python 5.sentiment/sent_analysis.py
5. Run topic modelling: python 6.Topic_analysis/topic_modeling.py
6. Generate figures and tables
    python 7.1plots/plot_sentiment_density_by_period.py
    python 7.1plots/plot_topic_by_time.py
    python 7.1plots/weekly_protest_peaks.py
    python 7.1plots/protest_by_year.py
    python 7.1plots/protest_covid_period.py
    python 7.1plots/vader_covid.py
    
    
    

        
