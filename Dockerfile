FROM continuumio/anaconda3:4.4.0

ENV PYTHONUNBUFFERED 0

RUN apt-get update && apt-get -y install \
    libgl1-mesa-glx

# build matplotlib font cache
RUN python -c 'import matplotlib.pyplot'

ADD ./requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

ADD . /src
WORKDIR /src
RUN python setup.py install
ENTRYPOINT ["glia"]
