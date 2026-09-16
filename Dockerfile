FROM python:3.11-slim

# Create standard non-root user with UID 1000 for Hugging Face Spaces
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    BACKEND=torch \
    MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

# Install Python requirements
COPY --chown=user requirements-spaces.txt /home/user/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source tree
COPY --chown=user . /home/user/app

EXPOSE 7860

CMD ["python3", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
