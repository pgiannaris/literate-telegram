# Health/Vaccine Hesitancy Sentiment Scraper

Collects text from Reddit and other websites (via RSS + article scraping), then labels each item for:
- vaccine/health hesitancy stance
- sentiment polarity

Uses local Ollama for classification.

## What it does
- Scrapes Reddit posts from selected subreddits + queries
- Pulls "other websites" from Google News RSS (and optional custom RSS feeds)
- Extracts article text
- Runs local LLM classification through Ollama
- Saves machine-readable output (`jsonl`, `csv`)

## Quick start
1. Create a Python env and install dependencies:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Make sure Ollama is running and a model is available:
```powershell
ollama pull llama3.1:8b
ollama serve
```

3. Run:
```powershell
python scraper.py
```

Outputs are written to `output/`.

## Usage
```powershell
python scraper.py `
  --queries "vaccine hesitancy" "health hesitancy" `
  --subreddits vaccines medicine Coronavirus conspiracy `
  --max-reddit-per-query 40 `
  --max-news-per-query 20 `
  --rss-feeds-file feeds.example.txt `
  --extra-urls-file urls.example.txt `
  --ollama-model llama3.1:8b `
  --output-dir output
```

## Optional inputs
- `--rss-feeds-file feeds.txt`
  - one RSS feed URL per line
- `--extra-urls-file urls.txt`
  - one direct webpage URL per line
- `--skip-ollama`
  - collect data only (no sentiment labeling)

## Output files
- `output/raw_records.jsonl`: collected text before labeling
- `output/labeled_records.jsonl`: records with `stance`, `sentiment`, `confidence`, `reason`
- `output/summary.csv`: grouped counts by source + stance + sentiment

## Notes
- Respect each site's Terms of Service and robots policies.
- Reddit's anonymous endpoints can rate-limit; this script uses retries and delay, but limits can still apply.
- For production use, consider official APIs for every source.
