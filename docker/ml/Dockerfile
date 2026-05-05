FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas==2.2.0 \
    scikit-learn==1.4.0 \
    xgboost==2.0.3 \
    numpy==1.26.0 \
    mlxtend==0.23.1 \
    matplotlib==3.8.0

COPY ml/   ./ml/
COPY utils/ ./utils/

CMD ["python", "-m", "ml.pipeline"]