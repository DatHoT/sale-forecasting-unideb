FROM python:3.11-slim

WORKDIR /srv/sale_forcasting_externaleffects_service

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY models ./models
COPY data ./data
RUN mkdir -p outputs

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
