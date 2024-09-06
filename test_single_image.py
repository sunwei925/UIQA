import os, argparse
import numpy as np
import torch
from torchvision import transforms

import models.UIQA as UIQA
from PIL import Image

from scipy.optimize import curve_fit
from scipy import stats
import pandas as pd
import random

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def performance_fit(y_label, y_output):
    y_output_logistic = fit_function(y_label, y_output)
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_output_logistic-y_label) ** 2).mean())

    return PLCC, SRCC, KRCC, RMSE

def get_spatial_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)
        
    if random_upsample:

        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=randratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)



    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    # target_videos = []

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                    :, t_s:t_e, h_so:h_eo, w_so:w_eo
                ]
    # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
    # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
    # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
    return target_video

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="Authentic Image Quality Assessment")
    parser.add_argument('--model_path', help='Path of model snapshot.', type=str)
    parser.add_argument('--trained_model_file', type=str)
    parser.add_argument('--popt_file', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--n_fragment', type=int, default=12)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--resize', type=int)
    parser.add_argument('--salient_patch_dimension', type=int, default=384)
    parser.add_argument('--crop_size', help='crop_size.',type=int)
    parser.add_argument('--gpu_ids', type=list, default=None)


    args = parser.parse_args()

    return args


if __name__ == '__main__':

    random_seed = 2
    torch.manual_seed(random_seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    args = parse_args()

    model_path = args.model_path
    popt_file = args.popt_file


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the network
    model_file = args.trained_model_file

    if args.model == 'UIQA':
        model = UIQA.UIQA_Model()

    # model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, model_file)))
    popt = np.load(os.path.join(model_path, popt_file))
    model.eval()

    
    transform_asethetics = transforms.Compose([transforms.Resize(args.resize),
                                               transforms.CenterCrop(args.crop_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])




    transform_distortion = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_distortion_preprocessing = transforms.Compose([transforms.ToTensor()])


    transform_saliency = transforms.Compose([
        transforms.CenterCrop(args.salient_patch_dimension),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    test_image = Image.open(os.path.join(args.image_path))

    test_image = test_image.convert('RGB')
    test_image_aesthetics = transform_asethetics(test_image)
    test_image_saliency = transform_saliency(test_image)

    test_image_distortion = transform_distortion_preprocessing(test_image)
    test_image_distortion = test_image_distortion.unsqueeze(1)
    test_image_distortion = get_spatial_fragments(
        test_image_distortion,
        fragments_h=args.n_fragment,
        fragments_w=args.n_fragment,
        fsize_h=32,
        fsize_w=32,
        aligned=32,
        nfrags=1,
        random=False,
        random_upsample=False,
        fallback_type="upsample"
    )
    test_image_distortion = test_image_distortion.squeeze(1)
    test_image_distortion = transform_distortion(test_image_distortion)

    test_image_aesthetics = test_image_aesthetics.unsqueeze(0)
    test_image_distortion = test_image_distortion.unsqueeze(0)
    test_image_saliency = test_image_saliency.unsqueeze(0)

    with torch.no_grad():
        test_image_aesthetics = test_image_aesthetics.to(device)
        test_image_saliency = test_image_saliency.to(device)
        test_image_distortion = test_image_distortion.to(device)
        outputs = model(test_image_aesthetics, test_image_distortion, test_image_saliency)
        score = outputs.item()
        print('The quality of the test image is {:.4f}'.format(score))