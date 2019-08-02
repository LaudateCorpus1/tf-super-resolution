FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends wget

RUN pip3 install \
    scikit-image==0.13.0 \
    scipy==0.19.1 \
    Pillow \
    ai-integration==1.0.11
COPY . /tf-perceptual-eusr

WORKDIR /tf-perceptual-eusr

RUN cd test && wget -nc https://s3-us-west-2.amazonaws.com/deepai-opensource-codebases-models/tf-super-resolution/4pp_eusr_pirm.pb

CMD []
ENTRYPOINT ["python3", "test/entrypoint.py"]
