FROM scratch

# dummy file to trigger a build on docker hub for jupyter-neuro

ADD hello /
CMD ["/hello"]
