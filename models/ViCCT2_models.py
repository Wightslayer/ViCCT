from timm.models.swin_transformer import SwinTransformer, _create_swin_transformer
import torch
import torch.nn as nn
from functools import partial


class ViCCTRegressionHead(nn.Module):
    #     def __init__(self, crop_size, embed_dim, init_weights=None):
    def __init__(self, crop_size, embed_dim):
        super().__init__()

        self.regression_head = nn.ModuleDict({
            'lin_scaler': nn.Sequential(
                nn.Linear(embed_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024)
            ),
            'folder': nn.Fold((crop_size, crop_size), kernel_size=32, stride=32)
        })

    #         if init_weights:
    #             self.regression_head['lin_scaler'].apply(init_weights)

    def forward(self, pre_den):
        pre_den = self.regression_head['lin_scaler'](pre_den)
        pre_den = pre_den.transpose(1, 2)
        den = self.regression_head['folder'](pre_den)

        return den


class DistilledRegressionTransformer(nn.Module):
    def __init__(self, base_model, **kwargs):
        super().__init__()

        self.base_model = nn.Sequential(*list(base_model.children())[:-2])  # Remove classification head and flattener
        #         n_parameters = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        #         print(n_parameters)

        self.regression_head = ViCCTRegressionHead(224, kwargs['embed_dim'] * 8)

    def forward(self, x):
        x = self.base_model(x)
        den = self.regression_head(x)
        return den


def swin_tiny_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k
    """

    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)

    base_model = _create_swin_transformer('swin_tiny_patch4_window7_224', pretrained=pretrained, **model_kwargs)

    return DistilledRegressionTransformer(base_model, **model_kwargs)


def swin_small_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k
    """

    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    base_model = _create_swin_transformer('swin_small_patch4_window7_224', pretrained=pretrained, **model_kwargs)
    full_model = DistilledRegressionTransformer(base_model, **model_kwargs)
    full_model.crop_size = 224
    
    return full_model