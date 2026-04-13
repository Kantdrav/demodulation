# Train and Deploy Guide

## 1) Install backend/training dependencies

```bash
cd /home/kantdravi/Desktop/Demod
/usr/bin/python3.10 -m venv .venv310
/home/kantdravi/Desktop/Demod/.venv310/bin/pip install --upgrade pip
/home/kantdravi/Desktop/Demod/.venv310/bin/pip install -r backend/requirements.txt
/home/kantdravi/Desktop/Demod/.venv310/bin/pip install -r training/requirements.txt
```

## 2) Train model on your dataset

Dataset path detected:
- /home/kantdravi/Downloads/dataset

Run full training:

```bash
cd /home/kantdravi/Desktop/Demod
/home/kantdravi/Desktop/Demod/.venv310/bin/python training/train_cnn.py \
  --dataset-dir "/home/kantdravi/Downloads/dataset" \
  --output-model backend/model.h5 \
  --output-config backend/model_config.json \
  --epochs 20 \
  --batch-size 16
```

Quick smoke test (small subset):

```bash
cd /home/kantdravi/Desktop/Demod
/home/kantdravi/Desktop/Demod/.venv310/bin/python training/train_cnn.py \
  --dataset-dir "/home/kantdravi/Downloads/dataset" \
  --output-model backend/model.h5 \
  --output-config backend/model_config.json \
  --epochs 2 \
  --batch-size 8 \
  --max-files-per-class 30
```

## 3) Run API locally

```bash
cd /home/kantdravi/Desktop/Demod/backend
/home/kantdravi/Desktop/Demod/.venv310/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## 4) Deploy to Render

1. Push the full project to GitHub.
2. In Render, create a Web Service from your repo.
3. Set root directory to backend.
4. Render uses backend/render.yaml automatically.
5. Ensure backend/model.h5 and backend/model_config.json exist in repo (or download them during build).

## 5) Connect Flutter

Set your API URL in flutter_app/lib/main.dart.

Example:
- https://your-service-name.onrender.com

Then run Flutter app as usual.
