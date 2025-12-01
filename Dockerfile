FROM nvidia/cuda:12.1.105-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip python3-dev git cmake build-essential libgl1-mesa-glx curl wget unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
COPY app.py .
COPY model/ ./model/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python3", "app.py"]