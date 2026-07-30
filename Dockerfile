# co2dash — container for the Streamlit app.
# Build:  docker build -t co2dash .
# Run:    docker run -p 8501:8501 --env-file .env co2dash
# (API tokens, e.g. ELECTRICITYMAPS_TOKEN, come from --env-file, never the image.)
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY examples ./examples

RUN pip install --no-cache-dir -e ".[ui,connectors]"

EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# healthcheck: Streamlit exposes /_stcore/health
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py"]
