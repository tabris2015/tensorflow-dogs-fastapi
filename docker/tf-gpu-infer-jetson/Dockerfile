FROM nvcr.io/nvidia/l4t-tensorflow:r32.5.0-tf2.3-py3

# RUN pip install tensorflow-datasets matplotlib pandas

ENV PORT=8000
COPY requirements-jetson.txt /
RUN pip3 install -r requirements-jetson.txt
COPY ./app /app
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
