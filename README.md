# Crop Disease Prediction MLOps Project

## Tech Stack
- PyTorch
- Flask
- Streamlit
- DVC
- Docker
- GitHub Actions
- AWS

## Run Flask API

```bash
python app/flask_app.py


streamlit run app/streamlit_app.py



python train.py
---

# 5. Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]