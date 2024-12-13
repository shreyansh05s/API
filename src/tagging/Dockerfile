# tagging/Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# install git
RUN apt-get update && apt-get install -y git

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install python-multipart

# Copy the tagging.py
COPY class_labels_indices.csv .
COPY tagging.py .

# Expose the API on port 8000
EXPOSE 8000

# Run the uvicorn server
CMD ["uvicorn", "tagging:app", "--host", "0.0.0.0", "--port", "8000"]
