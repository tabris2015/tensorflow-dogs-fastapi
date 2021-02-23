FROM python:3.6-slim
ENV PORT=8000
COPY requirements.txt /
RUN pip install -r requirements.txt
COPY ./app /app
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT