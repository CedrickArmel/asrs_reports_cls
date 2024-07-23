ARG BASE_VERSION=20.04
ARG PYTHONVERSION=3.10.13
ARG DEVICE=cpu

FROM drxc/python:${PYTHONVERSION}-ubuntu${BASE_VERSION}-${DEVICE}

ARG PYTHONVERSION=3.10.13

LABEL Author="https://github.com/CedrickArmel"

COPY . ${HOME}/app
WORKDIR ${HOME}/app
RUN set -eux ; \
    \
    source ${HOME}/.bashrc ; \
    pyenv virtualenv ${PYTHONVERSION} myenv ; \
    pyenv activate myenv ; \
    poetry install --without dev ; \
    pyenv global myenv ;

WORKDIR ${HOME}/app

ENV PYTHONPATH=/

ENTRYPOINT [ "/bin/bash" ]
