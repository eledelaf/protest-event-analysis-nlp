# Applying NLP to Protest Event Analysis
### A Case Study of UK Media Before, During and After COVID-19

An end-to-end NLP pipeline that automates Protest Event Analysis (PEA) on UK news coverage. The system collects articles from MediaCloud, scrapes full text, classifies protest relevance using a zero-shot transformer, and analyses sentiment and topics across outlets and time periods.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat&logo=mongodb&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)

---

## Objective

Protest Event Analysis traditionally requires manual coding of newspaper articles — a process that is slow and hard to scale. This project builds a reproducible, automated pipeline to:

- **Detect** which articles report concrete protest events (zero-shot classification)
- **Measure** the tone of protest coverage across outlets (VADER sentiment)
- **Discover** recurring themes and how they shift over time (BERTopic topic modelling)
- **Compare** coverage patterns before, during, and after the COVID-19 pandemic

**Outlets:** The Guardian (centre-left), Daily Mail (right-wing), Evening Standard (London-focused)
**Period:** January 2020 – December 2024

> This project was submitted as an MSc dissertation at Birkbeck, University of London. The full dissertation is available in [`TFM_final.pdf`](TFM_final.pdf).

---

## Pipeline Architecture

```
MediaCloud (59,742 candidate URLs)
 └── 1. Filter by outlet & date range         → 12,506 URLs
       └── 2. Scrape full article text         → 12,496 articles stored in MongoDB
             └── 3. Zero-shot classification   → 3,046 PROTEST / 9,256 NOT PROTEST
                   ├── 4. Sentiment analysis   → VADER compound scores
                   ├── 5. Topic modelling      → BERTopic clusters
                   └── 6. Temporal analysis    → Trends, peaks, COVID period comparisons
```

---

## Key Findings

### Corpus Scale
| Metric | Value |
|---|---|
| Articles scraped | **12,496** |
| Classified as PROTEST | **3,046** (24.4%) |
| Outlets analysed | **3** |
| Time span | **5 years** (2020–2024) |

### Coverage Peaks

Protest coverage is not spread evenly — a small number of high-attention events drive most of the volume:

- **Black Lives Matter (May–Jun 2020):** 225 articles in 3 weeks, triggered by the killing of George Floyd
- **US Capitol riot (Jan 2021):** 275 articles in 4 weeks, the single largest spike in the dataset
- **UK anti-immigration unrest (Jul–Aug 2024):** 102 articles in 2 weeks, following the Southport stabbings

### Sentiment

Protest reporting is overwhelmingly negative across all outlets (median VADER compound near -1.0). However, this reflects the language of conflict and disruption in news writing — not necessarily hostility toward protesters.

![Sentiment distribution by COVID period](7.2figures/sentiment_density_by_period.png)

### Topic Evolution

BERTopic identified 11–13 recurring themes. The dominant topics align with the coverage peaks above (BLM, Capitol, anti-immigration), while smaller clusters capture COVID restriction protests, Gaza/Ukraine-related mobilisation, and climate activism.

![Topic evolution over time](7.2figures/topic_by_time.png)

### Classifier Performance

The zero-shot model (facebook/bart-large-mnli) at threshold τ=0.65 achieves **52% precision and 48% recall** on a manually labelled sample (N=85). This is sufficient as a scalable filter for corpus construction, but not a replacement for human validation — consistent with current NLP literature on automated PEA.

---

## Tech Stack

| Component | Tools |
|---|---|
| Data collection | MediaCloud API, newspaper3k, requests |
| Storage | MongoDB, pymongo |
| Classification | Hugging Face Transformers (BART zero-shot) |
| Sentiment | VADER (nltk) |
| Topic modelling | BERTopic, sentence-transformers, scikit-learn |
| Analysis & plotting | pandas, numpy, matplotlib |

---

## Repository Structure

```
├── 1.MediaCloud/          # Raw MediaCloud CSV exports
├── 2.Data_cleaning/       # URL filtering and cleaning
├── 3.web_scrapping/       # Article scraping → MongoDB
├── 4.class_hf/            # Zero-shot protest classifier + threshold tuning
├── 5.sentiment/           # VADER sentiment analysis
├── 6.Topic_analysis/      # BERTopic modelling + outputs (CSV/HTML)
├── 7.1.plots/             # Analysis and plotting scripts
├── 7.2figures/            # Output figures
├── 7.3outputs/            # Output tables (CSV)
├── 8.MongoDB/             # Database export (.jsonl.gz)
├── TFM_final.pdf          # Full MSc dissertation
├── requirements.txt       # Python dependencies
└── README.md
```

<details>
<summary><b>Detailed file listing</b></summary>

### 1.MediaCloud
- `URLS.csv` — Raw CSV export from MediaCloud

### 2.Data_cleaning
- `FirstQuery.py` — Filters export to target outlets and date range → produces `URLs_clean.csv`

### 3.web_scrapping
- `fun_scrap3.py` — Scraping helper functions (newspaper3k)
- `scrape_to_mongo.py` — Reads cleaned URLs, scrapes text, upserts into MongoDB
- `URLs_clean.csv` — Filtered URL list

### 4.class_hf
- `hf_class.py` — Zero-shot classification functions
- `run_hf.py` — Runs classifier over MongoDB collection
- `threshold.py` — Threshold sweep and optimisation (maximises F0.5)
- `hf_results.py` — Classification output analysis
- `human_sample.csv` — Manually labelled evaluation sample (N=85)

### 5.sentiment
- `sent_analysis.py` — VADER sentiment scoring, writes results back to MongoDB

### 6.Topic_analysis
- `topic_modeling.py` — BERTopic modelling
- `topic_modeling/` — Output CSVs and interactive HTML visualisations

### 7.1.plots
- Scripts for sentiment density, topic trends, protest counts by year/period, and weekly peak detection

### 7.2figures
- `sentiment_density_by_period.png` — Sentiment distribution by COVID period
- `topic_by_time.png` — Topic group evolution over time

### 7.3outputs
- CSV tables: protest counts, sentiment means, weekly counts — all by outlet and COVID period

### 8.MongoDB
- `Texts.jsonl.gz` / `sample_texts.jsonl.gz` — Compressed database exports

</details>

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/eledelaf/Project-Master-Final-.git
cd Project-Master-Final-

# 2. Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure MongoDB access
#    Create a .env file with your MongoDB URI:
#    MONGO_URI=mongodb+srv://...

# 4. Run the pipeline in order
python 2.Data_cleaning/FirstQuery.py
python 3.web_scrapping/scrape_to_mongo.py
python 4.class_hf/run_hf.py
python 5.sentiment/sent_analysis.py
python 6.Topic_analysis/topic_modeling.py

# 5. Generate figures and analysis tables
python 7.1.plots/plot_sentiment_density_by_period.py
python 7.1.plots/plot_topic_by_time.py
python 7.1.plots/weekly_protest_peaks.py
python 7.1.plots/protest_by_year.py
python 7.1.plots/protest_covid_period.py
python 7.1.plots/vader_covid.py
```

> **Note:** The full pipeline requires MongoDB access and API keys. Pre-computed outputs (figures, CSV tables, and database exports) are included in the repo for review without running the code.
