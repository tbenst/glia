FROM alpine

# dummy file to trigger a build on docker hub for jupyter-neuro
ADD . /src
CMD ["echo dummy image! use tbenst/jupyter-neuro"]
