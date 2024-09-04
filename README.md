# UIQA

![visitors](https://visitor-badge.laobi.icu/badge?page_id=sunwei925/UIQA) [![](https://img.shields.io/github/stars/sunwei925/UIQA)](https://github.com/sunwei925/UIQA)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.13%2B-brightgree?logo=PyTorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sunwei925/UIQA)
[![arXiv](https://img.shields.io/badge/build-paper-red?logo=arXiv&label=arXiv)](https://arxiv.org/abs/2409.00749)

Official Code for **[Assessing UHD Image Quality from Aesthetics, Distortions, and Saliency](https://arxiv.org/abs/2409.00749)**



### TODO 
- [ ] release the model weights trained on the UHD-IQA dataset
- [ ] release the inference code

## Introduction
> **UHD images**, typically with resolutions equal to or higher than 4K, pose a significant challenge for efficient image quality assessment (IQA) algorithms, as adopting full-resolution images as inputs leads to overwhelming computational complexity and commonly used pre-processing methods like resizing or cropping may cause substantial loss of detail. To address this problem, we design a multi-branch deep neural network (DNN) to assess the quality of UHD images from three perspectives: **global aesthetic characteristics, local technical distortions, and salient content perception**. Specifically, *aesthetic features are extracted from low-resolution images downsampled from the UHD ones*, which lose high-frequency texture information but still preserve the global aesthetics characteristics. *Technical distortions are measured using a fragment image composed of mini-patches cropped from UHD images based on the grid mini-patch sampling strategy*. *The salient content of UHD images is detected and cropped to extract quality-aware features from the salient regions*. We adopt the Swin Transformer Tiny as the backbone networks to extract features from these three perspectives. The extracted features are concatenated and regressed into quality scores by a two-layer multi-layer perceptron (MLP) network. We employ the mean square error (MSE) loss to optimize prediction accuracy and the fidelity loss to optimize prediction monotonicity. Experimental results show that the proposed model achieves the best performance on the UHD-IQA dataset while maintaining the lowest computational complexity, demonstrating its effectiveness and efficiency. Moreover, the proposed model won **first prize in ECCV AIM 2024 UHD-IQA Challenge**.


## Image Pre-processing
![Image Pre-processing Figure](./figures/UHD_Image_Preprecessing.PNG)

> The different image pre-processing methods for UHD images. (a) is the proposed method, which utilizes the resized image, the fragment image, and the salient patch to extract features of aesthetic, distortion, and salient content. (b) samples all non-overlapped image patches for feature extraction. (c) selects three representative patches with the highest texture complexity for feature extraction.

## Model
![Model Figure](./figures/framework_UHD_IQA.PNG)

> The diagram of the proposed model. It consists of three modules: the image pre-processing module, the feature extraction module, and the quality regression module. We assess the quality of UHD images from three perspectives: global aesthetic characteristics, local technical distortions, and salient content perception, which are evaluated by the aesthetic assessment branch, distortion measurement branch, and salient content perception branch, respectively.

<!-- ## Computationl Complexity
![Computationl Complexity](./figures/macs.PNG) -->


## Usage
### Environments
- Requirements:
```
torch(>=1.13), torchvision, pandas, ptflops, numpy, Pillow
```
- Create a new environment
```
conda create -n UIQA python=3.8
conda activate UIQA 
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia # this command install pytorch version of 2.40, you can install pytorch >=1.13
pip install pandas ptflops numpy
```

### Dataset
Download the [UHD-IQA dataset](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html).

### Train UIQA

Download the [pre-trained model](https://www.dropbox.com/scl/fi/dk6co7hqquxpuq1nh04gf/Model_SwinT_AVA_epoch_10.pth?rlkey=tp13fdewe7hdosc3dja6al2dx&st=rg7tsy3t&dl=0) on AVA.

```
CUDA_VISIBLE_DEVICES=0,1 python -u train.py \
--num_epochs 100 \
--batch_size 12 \
--n_fragment 15 \
--resize 512 \
--crop_size 480 \
--salient_patch_dimension 480 \
--lr 0.00001 \
--lr_weight_L2 0.1 \
--lr_weight_pair 1 \
--decay_ratio 0.9 \
--decay_interval 10 \
--random_seed 1000 \
--snapshot ckpts \
--pretrained_path ckpts/Model_SwinT_AVA_size_480_epoch_10.pth \
--database_dir UHDIQA/challenge/training/ \
--model UIQA \
--multi_gpu True \
--print_samples 20 \
--database UHD_IQA \
>> logfiles/train_UIQA.log
```


