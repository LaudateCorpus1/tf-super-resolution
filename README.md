# Super Resolution: 4PP-EUSR

<p>
    <a href="https://cloud.docker.com/u/deepaiorg/repository/docker/deepaiorg/tf-super-resolution">
        <img src='https://img.shields.io/docker/cloud/automated/deepaiorg/tf-super-resolution.svg?style=plastic' />
        <img src='https://img.shields.io/docker/cloud/build/deepaiorg/tf-super-resolution.svg' />
    </a>
</p>

This model has been integrated with [ai_integration](https://github.com/deepai-org/ai_integration/blob/master/README.md) for seamless portability across hosting providers.

Four-pass perceptual super-resolution with enhanced upscaling

# Overview

This repository contains a TensorFlow-based implementation of **[4PP-EUSR ("Deep learning-based image super-resolution considering quantitative and perceptual quality")](http://arxiv.org/abs/1809.04789)**, which considers both the quantitative (e.g., PSNR) and perceptual quality (e.g., NIQE) of the upscaled images.
Our method **won the 2nd place and got the highest human opinion score for Region 2** in the [2018 PIRM Challenge on Perceptual Image Super-resolution at ECCV 2018](https://arxiv.org/abs/1809.07517).

Nvidia-Docker is required to run this image.

# For details see [Super Resolution](https://deepai.org/machine-learning-model/torch-srgan) on [Deep AI](https://deepai.org)

# Quick Start

docker pull deepaiorg/tf-super-resolution

### HTTP
```bash
nvidia-docker run --rm -it -e MODE=http -p 5000:5000 deepaiorg/tf-super-resolution
```
Open your browser to localhost:5000 (or the correct IP address)

### Command Line

Save your image as content.jpg in the current directory.
```bash
nvidia-docker run --rm -it -v `pwd`:/shared -e MODE=command_line deepaiorg/tf-super-resolution --image /shared/content.jpg --output /shared/output.jpg
```
# Docker build
```bash
docker build -t tf-super-resolution .
```

## Introduction

![BSD100 - 37073](figures/bsd100_37073.png)
※ The perceptual index is calculated by "0.5 * ((10 - [Ma](https://sites.google.com/site/chaoma99/sr-metric)) + [NIQE](https://doi.org/10.1109/LSP.2012.2227726))", which is used in the [PIRM Challenge](https://www.pirm2018.org/PIRM-SR.html). Lower is better.

Followings are the performance comparison evaluated on the [BSD100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) dataset.

![BSD100 PSNR vs. NIQE](figures/bsd100_psnr_niqe.png)

Method | PSNR (dB) (↓) | Perceptual Index
------------ | :---: | :---:
[EDSR](https://github.com/thstkdgus35/EDSR-PyTorch) | 27.796 | 5.326
[MDSR](https://github.com/thstkdgus35/EDSR-PyTorch) | 27.771 | 5.424
[EUSR](https://github.com/ghgh3269/EUSR-Tensorflow) | 27.674 | 5.307
[SRResNet-MSE](https://arxiv.org/abs/1609.04802) | 27.601 | 5.217
**4PP-EUSR (PIRM Challenge)** | **26.569** | **2.683**
[SRResNet-VGG22](https://arxiv.org/abs/1609.04802) | 26.322 | 5.183
[SRGAN-MSE](https://arxiv.org/abs/1609.04802) | 25.981 | 2.802
Bicubic interpolation | 25.957 | 6.995
[SRGAN-VGG22](https://arxiv.org/abs/1609.04802) | 25.697 | 2.631
[SRGAN-VGG54](https://arxiv.org/abs/1609.04802) | 25.176 | 2.351
[CX](https://arxiv.org/abs/1803.04626) | 24.581 | 2.250

Please cite following papers when you use the code, pre-trained models, or results:
- J.-H. Choi, J.-H. Kim, M. Cheon, J.-S. Lee: **Deep learning-based image super-resolution considering quantitative and perceptual quality**. arXiv:1809.04789 (2018) [[arXiv]](http://arxiv.org/abs/1809.04789)
```
@article{choi2018deep,
  title={Deep learning-based image super-resolution considering quantitative and perceptual quality},
  author={Choi, Jun-Ho and Kim, Jun-Hyuk and Cheon, Manri and Lee, Jong-Seok},
  journal={arXiv preprint arXiv:1809.04789},
  year={2018}
}
```
- J.-H. Kim, J.-S. Lee: **Deep residual network with enhanced upscaling module for super-resolution**. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 913-921 (2018) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Kim_Deep_Residual_Network_CVPR_2018_paper.html)
```
@inproceedings{kim2018deep,
  title={Deep residual network with enhanced upscaling module for super-resolution},
  author={Kim, Jun-Hyuk and Lee, Jong-Seok},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2018}
}
```
