FROM continuumio/anaconda3:4.0.0p0

# build matplotlib font cache
RUN python -c 'import matplotlib.pyplot'

ADD ./requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

ADD . /src
WORKDIR /src
RUN python setup.py install
ENTRYPOINT ["glia"]
