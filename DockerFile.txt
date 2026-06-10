FROM repos.esac.esa.int:62100/datalabs/jl_base:0.8.0-stable-24.04

COPY . /media/

RUN pip3 install --no-cache-dir -r /media/requirements.txt

RUN mkdir -p /media/notebooks

RUN mv /media/*.ipynb /media/notebooks/ 2>/dev/null || true

ENV PYTHONPATH=/media:$PYTHONPATH