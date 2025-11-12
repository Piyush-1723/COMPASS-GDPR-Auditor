## Use a newer PyTorch image built with CUDA 12.8 which includes support for compute capability sm_120 (RTX 50xx)
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Copy requirements first for caching (file lives in backend/ for build context)
COPY requirements.txt ./

# Install system fonts for Unicode PDF rendering and Python packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
	 fonts-dejavu-core \
	 libpango-1.0-0 \
	 libpangocairo-1.0-0 \
	 libpangoft2-1.0-0 \
	 libcairo2 \
	 libgdk-pixbuf2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy backend source code only (model weights are mounted via docker-compose volume for faster iteration)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
