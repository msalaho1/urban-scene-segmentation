FROM python:3.10-slim

# Installingg system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean

WORKDIR /app


COPY . /app


RUN pip install --no-cache-dir -r requirements.txt

VOLUME /app/output: ./output/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]