FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

RUN pip install tensorflow-datasets matplotlib pandas

WORKDIR /app

ENTRYPOINT [ "jupyter", "notebook" ]