# UIQA

![visitors](https://visitor-badge.laobi.icu/badge?page_id=sunwei925/UIQA) [![](https://img.shields.io/github/stars/sunwei925/UIQA)](https://github.com/sunwei925/UIQA)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.13%2B-brightgree?logo=PyTorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sunwei925/UIQA)
[![arXiv](https://img.shields.io/badge/build-paper-red?logo=arXiv&label=arXiv)](https://arxiv.org/abs/2409.00749)

Official Code for **[Assessing UHD Image Quality from Aesthetics, Distortions, and Saliency](https://arxiv.org/abs/2409.00749)**

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

## Performance
#### Compared with state-of-the-art IQA methods
- Performance on the validation set of the UHD-IQA dataset

| Methods | SRCC | PLCC | KRCC | RMSE | MAE |
| :---: | :---:| :---:|:---: |:---: |:---: |
|HyperIQA|0.524|0.182| 0.359|0.087| 0.055|
|Effnet-2C-MLSP|0.615| 0.627|0.445|0.060|0.050|
|CONTRIQUE|0.716| 0.712|0.521|0.049|0.038|
|ARNIQA|0.718|0.717|0.523| 0.050|0.039|
|CLIP-IQA+|0.743|0.732| 0.546| 0.108|0.087|
|QualiCLIP|0.757|0.752|0.557|0.079|0.064|
|**UIQA**|**0.817**| **0.823**| **0.625**|**0.040**| **0.032**|

- Performance on the test set of the UHD-IQA dataset

| Methods | SRCC | PLCC | KRCC | RMSE | MAE |
| :---: | :---:| :---:|:---: |:---: |:---: |
|HyperIQA|0.553| 0.103| 0.389|0.118|0.070 |
|Effnet-2C-MLSP|0.675|0.641 | 0.491|0.074|0.059|
|CONTRIQUE|0.732| 0.678|0.532| 0.073|0.052|
|ARNIQA|0.739|0.694|0.544|  0.052|0.739|
|CLIP-IQA+|0.747| 0.709| 0.551| 0.111| 0.089|
|QualiCLIP|0.770|0.725|0.570|0.083|0.066|
|**UIQA**|**0.846**|  **0.798**|**0.657**|**0.061**| **0.042**|










#### Performance on ECCV AIM 2024 UHD-IQA Challenge
| Team | SRCC | PLCC | KRCC | RMSE | MAE |
| :---: | :---:| :---:|:---: |:---: |:---: |
| **SJTU MMLab (ours)** | **0.846** | 0.798 | **0.657** |  **0.061** | **0.042** |
| CIPLAB | 0.835 | **0.800** |  0.642 |   0.064 | 0.044 |
| ZX AIE Vector | 0.795 | 0.768 | 0.605  | 0.062 | 0.044 |
| I2Group | 0.788 | 0.756 | 0.598  | 0.066 | 0.046 |
| Dominator | 0.731 | 0.712 | 0.539  | 0.072 |  0.052 |
|ICL|0.517| 0.521|0.361| 0.136| 0.115|

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

### Test UIQA
Put your trained model in the ckpts folder, or download the provided trained model ([model weights](https://www.dropbox.com/scl/fi/mgvvt902zehhmo6drnxve/UIQA.pth?rlkey=413edq08c8qnxbrlgclnq0va6&st=yzcskkus&dl=0), [quality alignment profile file](https://www.dropbox.com/scl/fi/1st2jjga6ssirvsex5oo6/UIQA.npy?rlkey=6mbf2utiz1t3dlm5nl635cvmz&st=n0tvbqv9&dl=0)) on the UHD-IQA dataset into the ckpts folder.

```
CUDA_VISIBLE_DEVICES=0 python -u test_single_image.py \
--model_path ckpts/ \
--trained_model_file UIQA.pth \
--popt_file UIQA.npy \
--image_path demo/8.jpg \
--resize 512 \
--crop_size 480 \
--n_fragment 15 \
--salient_patch_dimension 480 \
--model UIQA
```

## Citation
**If you find this code is useful for  your research, please cite**:

```latex
@article{sun2024assessing,
  title={Assessing UHD Image Quality from Aesthetics, Distortions, and Saliency},
  author={Sun, Wei and Zhang, Weixia and Cao, Yuqin and Cao, Linhan and Jia, Jun and Chen, Zijian and Zhang, Zicheng and Min, Xiongkuo and Zhai, Guangtao},
  journal={arXiv preprint arXiv:2409.00749},
  year={2024}
}
```

## Acknowledgement

1. <https://github.com/zwx8981/LIQE>
2. <https://github.com/VQAssessment/FAST-VQA-and-FasterVQA>
3. <https://github.com/imfing/ava_downloader>