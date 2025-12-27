import pandas as pd

"""
This python file does the first query getting from URLS gotten directly from 
MediaCloud to URLS_clean where I have the urls that I am going to use for the scraping
"""

# ---------------------------------------------------------------------
# 0. URL prefixes I want to exclude, non-news or irrelevant sections
# ---------------------------------------------------------------------
EXCLUDED_URL_PREFIXES = [
    # Daily Mail
    "https://www.dailymail.co.uk/news/royals",
    "https://www.dailymail.co.uk/sport",
    "https://www.dailymail.co.uk/tvshowbiz",
    "https://www.dailymail.co.uk/lifestyle",
    "https://www.dailymail.co.uk/health",
    "https://www.dailymail.co.uk/travel",
    "https://www.dailymail.co.uk/buyline",
    "https://www.dailymail.co.uk/femail",
    "https://www.dailymail.co.uk/galleries",
    "https://www.dailymail.co.uk/home",
    "https://www.dailymail.co.uk/tv",
    "https://www.dailymail.co.uk/stage",

    # The Guardian
    "https://www.theguardian.com/uk/commentisfree",
    "https://www.theguardian.com/commentisfree",
    "https://www.theguardian.com/uk/sport",
    "https://www.theguardian.com/sports",
    "https://www.theguardian.com/uk/culture",
    "https://www.theguardian.com/culture",
    "https://www.theguardian.com/uk/lifeandstyle",
    "https://www.theguardian.com/lifeandstyle",
    "https://www.theguardian.com/music",
    "https://www.theguardian.com/stage",
    "https://www.theguardian.com/food",
    "https://www.theguardian.com/film",
    "https://www.theguardian.com/books",
    "https://www.theguardian.com/travel",
    "https://www.theguardian.com/tv-and-radio",
    "https://www.theguardian.com/football",
    "https://www.theguardian.com/artanddesign",

    # Evening Standard
    "https://www.standard.co.uk/sport",
    "https://www.standard.co.uk/lifestyle",
    "https://www.standard.co.uk/culture",
    "https://www.standard.co.uk/going-out",
    "https://www.standard.co.uk/homesandproperty",
    "https://www.standard.co.uk/comment",
    "https://www.standard.co.uk/esmagazine",
    "https://www.standard.co.uk/going-out/restaurants",
    "https://www.standard.co.uk/reveller/restaurants",
    "https://www.standard.co.uk/escapist",
    "https://www.standard.co.uk/insider",
    "https://www.standard.co.uk/shopping",

]

def is_excluded_title(title):
    """
    This function determines if a title should be excluded.
    Excludes titles like "News Headlines", "Morning Headlines", "Evening Headlines", and "briefing:" patterns.
    """
    if not isinstance(title, str):
        return False
    t = title.strip().lower()

    # 1) "News Headlines | ..." or similar
    if t.startswith("news headlines"):
        return True
    if t.startswith("morning headlines"):
        return True
    if t.startswith("evening headlines"):
        return True

    # 2) "<weekday> briefing: ..." or any "briefing:" pattern
    if " briefing:" in t:
        return True
    
    if "photos of the day" in t:
        return True 
    
    return False


# ---------------------------------------------------------------------
# 1. Load raw URLs from MediaCloud
# ---------------------------------------------------------------------
file_path = "/Users/elenadelafuente/Desktop/MASTER/TFM/Project/Project-Master/2.Data_cleaning/URLS.csv"
df = pd.read_csv(file_path, sep=';', skiprows=1, header=None)
df = df.iloc[1:].reset_index(drop=True)

header = ["id", "indexed_date", "language", "media_name", "media_url", "publish_date", "title", "url"]
df.columns = header
print(df)

# ---------------------------------------------------------------------
# 2. Keep only selected newspapers
# ---------------------------------------------------------------------
# theguardian.com, standard.co.uk, dailymail.co.uk
df_clean = df[
    (df["media_url"] == "theguardian.com") |
    (df["media_url"] == "standard.co.uk") |
    (df["media_url"] == "dailymail.co.uk") 
]

# Keep only the columns I care about
df_clean = df_clean[["id", "media_url", "publish_date", "title", "url"]]

# ---------------------------------------------------------------------
# 3. Exclude URLs based on unwanted section prefixes
# ---------------------------------------------------------------------
# I drop any row where 'url' starts with one of the EXCLUDED_URL_PREFIXES
mask_keep = ~df_clean["url"].astype(str).str.startswith(tuple(EXCLUDED_URL_PREFIXES))
df_clean = df_clean[mask_keep]

# Reset index after filtering
df_clean = df_clean.reset_index(drop=True)

print(df_clean)

# ---------------------------------------------------------------------
# 4. Exclude Titles like:  News Headlines or Tuesday briefing
# ---------------------------------------------------------------------
# We drop any row where 'title' starts with one of the EXCLUDED_TITLES
mask_keep_title = ~df_clean["title"].apply(is_excluded_title)
df_clean = df_clean[mask_keep_title]

# ---------------------------------------------------------------------
# 5. Save cleaned URLs
# ---------------------------------------------------------------------
df_clean.to_csv("3.web_scrapping/URLs_clean.csv", sep=";", index=False)
