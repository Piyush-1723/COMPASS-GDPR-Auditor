## Use a newer PyTorch image built with CUDA 12.8 which includes support for compute capability sm_120 (RTX 50xx)
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python packages properly (do NOT reinstall torch; it's already in the base image with cu12.4)
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy app files
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
