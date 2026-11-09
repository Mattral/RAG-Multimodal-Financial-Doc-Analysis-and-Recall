# Local Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
sudo apt-get install -y poppler-utils tesseract-ocr
cp .env.example .env  # set OPENAI_API_KEY
filing-qa ingest tesla_10k.pdf --tenant demo
filing-qa query "What was Q3 revenue?" --show-sources
filing-qa serve
```
