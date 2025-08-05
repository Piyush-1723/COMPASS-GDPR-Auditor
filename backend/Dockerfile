# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the file that lists requirements
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during the build
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab

# Copy the rest of the application's code into the container
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run main.py when the container launches
# The --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]