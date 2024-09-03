import torch
import torch.nn as nn
import torchvision.models as models






class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Model_SwinT(nn.Module):
    def __init__(self):
        super(Model_SwinT, self).__init__()

        model = models.swin_t(weights='Swin_T_Weights.DEFAULT')
        model.head = Identity()

        # spatial quality analyzer
        self.feature_extraction = model

        # quality regressor
        self.quality = self.quality_regression(768, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x):

        x = self.feature_extraction(x)
        x = self.quality(x)
                    
        return x
























class UIQA_Model(torch.nn.Module):
    def __init__(self, pretrained_path=None):
        
        super(UIQA_Model, self).__init__()
        # Aesthetics feature extractor
        swin_t_aesthetics = Model_SwinT()
        if pretrained_path!=None:
            print('load aesthetics model')
            swin_t_aesthetics.load_state_dict(torch.load(pretrained_path))

        # Distortion feature extractor
        swin_t_distortion = Model_SwinT()
        if pretrained_path!=None:
            print('load distortion model')
            swin_t_distortion.load_state_dict(torch.load(pretrained_path))

        # Salient image feature extractor
        swin_t_salient = Model_SwinT()
        if pretrained_path!=None:
            print('load saliency model')
            swin_t_salient.load_state_dict(torch.load(pretrained_path))

        self.aesthetics_feature_extraction = swin_t_aesthetics.feature_extraction
        self.distortion_feature_extraction = swin_t_distortion.feature_extraction
        self.saliency_feature_extraction = swin_t_salient.feature_extraction
        self.quality = self.quality_regression(768+768+768, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x_aesthetics, x_distortion, x_saliency):

        x_aesthetics = self.aesthetics_feature_extraction(x_aesthetics)
        x_distortion = self.distortion_feature_extraction(x_distortion)
        x_saliency = self.saliency_feature_extraction(x_saliency)
        # fuse the features
        x = torch.cat((x_aesthetics, x_distortion, x_saliency), dim = 1)

        x = self.quality(x)
            
        return x