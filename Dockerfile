FROM python:3.11-slim

WORKDIR /app

# Only build-essential is needed for pydantic C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces serves on port 7860
EXPOSE 7860

CMD ["sh", "-c", "python inference.py --dry-run --task all --verbose && echo 'Simulation Complete. Starting server...' && uvicorn server.app:app --host 0.0.0.0 --port 7860"]
