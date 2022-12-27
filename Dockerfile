FROM continuumio/anaconda3:latest

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash

RUN apt-get update --yes && \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends\
    git\
    wget\
    curl\
    git\
    bash\
    openssh-server &&\
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN conda config --append channels conda-forge
RUN conda install -y cudatoolkit
RUN conda install -y libgcc gmp
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install git+https://github.com/ShivamShrirao/diffusers
RUN pip install accelerate==0.12.0 transformers ftfy bitsandbytes gradio natsort

# GLIBCXX_3.4.29 fix
RUN cp /opt/conda/lib/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/

ADD start.sh /

RUN chmod +x /start.sh

CMD [ "/start.sh" ]