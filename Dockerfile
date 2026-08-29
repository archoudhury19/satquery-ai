FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system C-libraries required by OpenCV and Rasterio/GDAL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create uploads and generated directories with full write permissions
RUN mkdir -p uploads generated backend/generated && \
    chmod -R 777 uploads generated backend/generated

EXPOSE 8000
EXPOSE 7860
EXPOSE 10000

CMD ["python", "app.py"]