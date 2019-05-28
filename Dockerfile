FROM nvidia/cuda:9.0-base-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip=8.1.1-2ubuntu0.4 \
        python3-dev=3.5.1-3 \
        python3-setuptools=20.7.0-1 \
        python3-tk=3.5.1-1 \
        wget \
        git=1:2.7.4-0ubuntu1.6 

RUN pip3 install -U pip

RUN pip3 install Pillow
RUN pip3 install tensorflow-gpu
RUN pip3 install tensorflow
RUN pip3 --no-cache-dir install \
    backports.weakref==1.0rc1 \
    bleach==1.5.0 \
    cycler==0.10.0 \
    decorator==4.1.2 \
    h5py==2.7.0 \
    html5lib==0.9999999 \
    Markdown==2.6.8 \
    matplotlib==2.0.2 \
    networkx==1.11 \
    numpy==1.13.3 \
    olefile==0.44 \
    protobuf==3.6.1 \
    pyparsing==2.2.0 \
    python-dateutil==2.6.1 \
    pytz==2017.2 \
    PyWavelets==0.5.2 \
    scikit-image==0.13.0 \
    scipy==0.19.1 \
    six==1.10.0 \
    Werkzeug==0.12.2
    

RUN pip3 install ai-integration

COPY . /tf-perceptual-eusr/test

WORKDIR /tf-perceptual-eusr/test

RUN cd test && wget -nc https://s3-us-west-2.amazonaws.com/deepai-opensource-codebases-models/tf-super-resolution/4pp_eusr_pirm.pb

ENTRYPOINT ["python3", "test/entrypoint.py"]
