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

# Default: smoke-test all three tasks without an LLM token.
# To run the full LLM baseline: docker run -e HF_TOKEN=hf_xxx <image> python inference.py --task all --verbose
CMD ["python", "inference.py", "--dry-run", "--task", "all", "--verbose"]
