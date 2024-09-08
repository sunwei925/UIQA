import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import random
import cv2




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



class AVA_dataloader_pair(Dataset):
    def __init__(self, data_dir, csv_path, transform, database, seed):
        self.database = database


        tmp_df = pd.read_csv(csv_path)
        image_name = tmp_df['image_num'].to_list()
        image_score = np.zeros([len(image_name)])
        for i_vote in range(1,11):
            image_score += i_vote * tmp_df['vote_'+str(i_vote)].to_numpy()

        n_images = len(image_name)
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(n_images)

        if 'train' in database:
            index_subset = index_rd[ : int(n_images * 0.8)]
            self.X_train = [str(image_name[i])+'.jpg' for i in index_subset]
            self.Y_train = [image_score[i] for i in index_subset]
        elif 'test' in database:
            index_subset = index_rd[int(n_images * 0.8) : ]
            self.X_train = [str(image_name[i])+'.jpg' for i in index_subset]
            self.Y_train = [image_score[i] for i in index_subset]
        else:
            raise ValueError(f"Unsupported subset database name: {database}")
        print(self.X_train)



        self.data_dir = data_dir
        self.transform = transform
        self.length = len(self.X_train)

    def __getitem__(self, index):

        index_second = random.randint(0, self.length - 1)
        if index == index_second:
            index_second = (index_second + 1) % self.length
        while self.Y_train[index] == self.Y_train[index_second]:
            index_second = random.randint(0, self.length - 1)
            if index == index_second:
                index_second = (index_second + 1) % self.length

        path = os.path.join(self.data_dir,self.X_train[index])
        path_second = os.path.join(self.data_dir,self.X_train[index_second])

        img = Image.open(path)
        img = img.convert('RGB')


        img_second = Image.open(path_second)
        img_second = img_second.convert('RGB')

        img_overall = self.transform(img)
        img_second_overall = self.transform(img_second)

        y_mos = self.Y_train[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))


        y_mos_second = self.Y_train[index_second]
        y_label_second = torch.FloatTensor(np.array(float(y_mos_second)))

        return img_overall, y_label, img_second_overall, y_label_second




class AVA_dataloader(Dataset):
    def __init__(self, data_dir, csv_path, transform, database, seed):
        self.database = database


        tmp_df = pd.read_csv(csv_path)
        image_name = tmp_df['image_num'].to_list()
        image_score = np.zeros([len(image_name)])
        for i_vote in range(1,11):
            image_score += i_vote * tmp_df['vote_'+str(i_vote)].to_numpy()

        n_images = len(image_name)
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(n_images)

        if 'train' in database:
            index_subset = index_rd[ : int(n_images * 0.8)]
            self.X_train = [str(image_name[i])+'.jpg' for i in index_subset]
            self.Y_train = [image_score[i] for i in index_subset]
        elif 'test' in database:
            index_subset = index_rd[int(n_images * 0.8) : ]
            self.X_train = [str(image_name[i])+'.jpg' for i in index_subset]
            self.Y_train = [image_score[i] for i in index_subset]
        else:
            raise ValueError(f"Unsupported subset database name: {database}")
        print(self.X_train)



        self.data_dir = data_dir
        self.transform = transform
        self.length = len(self.X_train)

    def __getitem__(self, index):

        path = os.path.join(self.data_dir,self.X_train[index])

        img = Image.open(path)
        img = img.convert('RGB')

        img_overall = self.transform(img)

        y_mos = self.Y_train[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))

        return img_overall, y_label


    def __len__(self):
        return self.length

class UIQA_dataloader_pair(Dataset):
    def __init__(self, data_dir, csv_path, transform, database, n_fragment=12, salient_patch_dimension=448, seed=0):
        self.database = database
        self.salient_patch_dimension = salient_patch_dimension
        self.n_fragment = n_fragment

        tmp_df = pd.read_csv(csv_path)
        image_name = tmp_df['image_name'].to_list()
        mos = tmp_df['quality_mos'].to_list()

        n_images = len(image_name)
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(n_images)

        if 'train' in database:
            index_subset = index_rd[ : int(n_images * 0.8)]
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos[i] for i in index_subset]
        elif 'test' in database:
            index_subset = index_rd[int(n_images * 0.8) : ]
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos[i] for i in index_subset]
        elif 'all' in database:
            index_subset = index_rd
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos[i] for i in index_subset]
        else:
            raise ValueError(f"Unsupported subset database name: {database}")
        print(self.X_train)









        self.data_dir = data_dir                
        self.transform_aesthetics = transform
        self.transform_distortion = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_distortion_preprocessing = transforms.Compose([transforms.ToTensor()])
        self.transform_saliency = transforms.Compose([
            transforms.CenterCrop(self.salient_patch_dimension),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.length = len(self.X_train)

    def __getitem__(self, index):

        index_second = random.randint(0, self.length - 1)
        if index == index_second:
            index_second = (index_second + 1) % self.length
        while self.Y_train[index] == self.Y_train[index_second]:
            index_second = random.randint(0, self.length - 1)
            if index == index_second:
                index_second = (index_second + 1) % self.length

        path = os.path.join(self.data_dir, self.X_train[index])
        path_second = os.path.join(self.data_dir, self.X_train[index_second])

        img = Image.open(path)
        img = img.convert('RGB')


        img_second = Image.open(path_second)
        img_second = img_second.convert('RGB')

        img_aesthetics = self.transform_aesthetics(img)
        img_second_aesthetics = self.transform_aesthetics(img_second)

        img_saliency = self.transform_saliency(img)
        img_second_saliency = self.transform_saliency(img_second)


        img_distortion = self.transform_distortion_preprocessing(img)
        img_second_distortion = self.transform_distortion_preprocessing(img_second)

        img_distortion = img_distortion.unsqueeze(1)
        img_second_distortion = img_second_distortion.unsqueeze(1)

        img_distortion = get_spatial_fragments(
            img_distortion,
            fragments_h=self.n_fragment,
            fragments_w=self.n_fragment,
            fsize_h=32,
            fsize_w=32,
            aligned=32,
            nfrags=1,
            random=False,
            random_upsample=False,
            fallback_type="upsample"
        )
        img_second_distortion = get_spatial_fragments(
            img_second_distortion,
            fragments_h=self.n_fragment,
            fragments_w=self.n_fragment,
            fsize_h=32,
            fsize_w=32,
            aligned=32,
            nfrags=1,
            random=False,
            random_upsample=False,
            fallback_type="upsample"
        )

        img_distortion = img_distortion.squeeze(1)
        img_second_distortion = img_second_distortion.squeeze(1)

        img_distortion = self.transform_distortion(img_distortion)
        img_second_distortion = self.transform_distortion(img_second_distortion)

        y_mos = self.Y_train[index]

        y_label = torch.FloatTensor(np.array(float(y_mos)))


        y_mos_second = self.Y_train[index_second]

        y_label_second = torch.FloatTensor(np.array(float(y_mos_second)))


        data = {'img_aesthetics': img_aesthetics,
                'img_distortion': img_distortion,
                'img_saliency': img_saliency,
                'y_label': y_label,
                'img_second_aesthetics': img_second_aesthetics,
                'img_second_distortion': img_second_distortion,
                'img_second_saliency': img_second_saliency,
                'y_label_second': y_label_second}

        return data


    def __len__(self):
        return self.length


class UIQA_dataloader(Dataset):
    def __init__(self, data_dir, csv_path, transform, database, n_fragment=12, salient_patch_dimension=448, seed=0):
        self.database = database
        self.salient_patch_dimension = salient_patch_dimension
        self.n_fragment = n_fragment


        tmp_df = pd.read_csv(csv_path)
        image_name = tmp_df['image_name'].to_list()
        mos = tmp_df['quality_mos'].to_list()

        n_images = len(image_name)
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(n_images)

        if 'train' in database:
            index_subset = index_rd[ : int(n_images * 0.8)]
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos[i] for i in index_subset]
        elif 'test' in database:
            index_subset = index_rd[int(n_images * 0.8) : ]
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos[i] for i in index_subset]
        elif 'all' in database:
            index_subset = index_rd
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos[i] for i in index_subset]
        else:
            raise ValueError(f"Unsupported subset database name: {database}")
        print(self.X_train)



        self.data_dir = data_dir
        self.transform_aesthetics = transform
        self.transform_distortion = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_distortion_preprocessing = transforms.Compose([transforms.ToTensor()])
        self.transform_saliency = transforms.Compose([
            transforms.CenterCrop(self.salient_patch_dimension),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.length = len(self.X_train)

    def __getitem__(self, index):
        path = os.path.join(self.data_dir,self.X_train[index])

        img = Image.open(path)
        img = img.convert('RGB')

        img_aesthetics = self.transform_aesthetics(img)
        img_saliency = self.transform_saliency(img)


        img_distortion = self.transform_distortion_preprocessing(img)
        img_distortion = img_distortion.unsqueeze(1)
        img_distortion = get_spatial_fragments(
            img_distortion,
            fragments_h=self.n_fragment,
            fragments_w=self.n_fragment,
            fsize_h=32,
            fsize_w=32,
            aligned=32,
            nfrags=1,
            random=False,
            random_upsample=False,
            fallback_type="upsample"
        )
        img_distortion = img_distortion.squeeze(1)
        img_distortion = self.transform_distortion(img_distortion)

        y_mos = self.Y_train[index]

        y_label = torch.FloatTensor(np.array(float(y_mos)))

        data = {'img_aesthetics': img_aesthetics,
                'img_distortion': img_distortion,
                'img_saliency': img_saliency,
                'y_label': y_label}

        return data


    def __len__(self):
        return self.length
