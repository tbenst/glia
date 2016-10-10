FROM continuumio/anaconda3:4.0.0p0

# build matplotlib font cache
RUN python -c 'import matplotlib.pyplot'

RUN pip install pyyaml
RUN pip install click

ADD . /src
RUN cd /src && python setup.py install
ENTRYPOINT ["glia"]
