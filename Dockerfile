FROM continuumio/anaconda3:4.0.0p0

# build matplotlib font cache
RUN python -c 'import matplotlib.pyplot'


ADD . /src
WORKDIR /src
RUN pip install -r requirements.txt
RUN python setup.py install
ENTRYPOINT ["glia"]
