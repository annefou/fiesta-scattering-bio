FROM python:3.11-slim

RUN pip install --no-cache-dir \
        "foscat @ git+https://github.com/annefou/FOSCAT.git@v0.1.0-cpu" \
        scikit-learn \
        numpy \
        matplotlib \
        Pillow \
        healpy

WORKDIR /app
COPY . /app

CMD ["python", "01_plankton_classification.py"]
