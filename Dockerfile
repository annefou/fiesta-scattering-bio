FROM python:3.11-slim

RUN pip install --no-cache-dir \
        foscat \
        scikit-learn \
        numpy \
        matplotlib \
        Pillow \
        healpy

WORKDIR /app
COPY . /app

CMD ["python", "01_plankton_classification.py"]
