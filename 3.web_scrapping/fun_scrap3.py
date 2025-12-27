from newspaper import Article, Config

def scrape_and_text(url, filename=None):
    """
    Scrapes only the article body using Newspaper3k.
    """
    try:
        # Configuration to make the request look like a real browser
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
        config.request_timeout = 10

        article = Article(url, config=config)
        article.download()
        article.parse()
        text = article.text

        # If text is too short, it might be a failed scrape or a video page
        if not text or len(text) < 100:
            print(f"[Newspaper3k] Content too short or empty: {url}")
            return None

        return text

    except Exception as e:
        print(f"[Newspaper3k] Error processing {url}: {e}")
        return None
    