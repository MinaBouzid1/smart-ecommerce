FROM python:3.10-slim

WORKDIR /app

# Dépendances système pour Playwright
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir \
    requests==2.31.0 \
    beautifulsoup4==4.12.2 \
    scrapy==2.11.0 \
    playwright==1.42.0 \
    pandas==2.2.0 \
    fake-useragent==1.4.0

# Navigateur Playwright
RUN playwright install chromium

# Code source
COPY agents/ ./agents/
COPY utils/  ./utils/
COPY main.py .

CMD ["python", "main.py", "--mode", "scraping"]