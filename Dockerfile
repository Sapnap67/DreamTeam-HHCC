FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_DATA_DIR=/tmp/blindspot-guardian \
    YOLO_MODEL_PATH=/app/yolo11n.pt \
    POSE_MODEL_PATH=/app/models/pose_landmarker_lite.task \
    MAX_UPLOAD_BYTES=104857600

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-cloud.txt ./
RUN python -m pip install --no-cache-dir -r requirements-cloud.txt

COPY . .
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" \
    && python download_pose_model.py

EXPOSE 10000
CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 0 --bind 0.0.0.0:${PORT:-10000} app:app"]

