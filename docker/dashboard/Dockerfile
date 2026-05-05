FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit==1.32.0 \
    pandas==2.2.0 \
    plotly==5.20.0 \
    numpy==1.26.0

COPY dashboard/ ./dashboard/
COPY data/       ./data/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]