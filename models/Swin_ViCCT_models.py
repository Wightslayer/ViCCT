from timm.models.swin_transformer import SwinTransformer, _create_swin_transformer
import torch
import torch.nn as nn
from timm.models.registry import register_model


def init_model_state(model, init_path):
    if init_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            init_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(init_path, map_location='cpu')
    if 'model' in checkpoint:
        pretrained_state = checkpoint['model']
    else:
        pretrained_state = checkpoint
        print('@'*1000)  # Does this still happen?

    model_state_dict = model.state_dict()
    # With this, we are able to load the pretrained modules while ignoring the new regression modules.
    for key in list(model_state_dict.keys())[:-4]:  # Head and norm don't match when we remove last block
        if key in pretrained_state:
            model_state_dict[key] = pretrained_state[key]
        else:
            print(f'Key: {key} note in state dict!')

    model.load_state_dict(model_state_dict)

    return model


def load_pretrained(model, init_path):
    """ Loads a pretrained crowd counting network. """

    resume_state = torch.load(init_path)
    model.load_state_dict(resume_state['net'])

    return model


class ViCCTRegressionHead(nn.Module):
    #     def __init__(self, crop_size, embed_dim, init_weights=None):
    def __init__(self, crop_size, embed_dim):
        super().__init__()

        self.regression_head = nn.ModuleDict({
            'lin_scaler': nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            ),
            'folder': nn.Fold((crop_size, crop_size), kernel_size=16, stride=16)
        })

    def forward(self, pre_den):
        pre_den = self.regression_head['lin_scaler'](pre_den)
        pre_den = pre_den.transpose(1, 2)
        den = self.regression_head['folder'](pre_den)

        return den


class DistilledRegressionTransformer(nn.Module):
    def __init__(self, base_model, **kwargs):
        super().__init__()

        self.base_model = nn.Sequential(*list(base_model.children())[:-2])  # Remove classification head and flattener
        self.regression_head = ViCCTRegressionHead(224, kwargs['embed_dim'] * 4)

    def forward(self, x):
        x = self.base_model(x)
        den = self.regression_head(x)
        return den


# def Swin_ViCCT_tiny(pretrained_path=None, **kwargs):
#     """ Swin-T @ 224x224, trained ImageNet-1k
#     """
#
#     model_kwargs = dict(
#         patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
#
#     base_model = _create_swin_transformer('swin_tiny_patch4_window7_224', pretrained=False, **model_kwargs)
#     print("MODEL NOT SUPPORTED YET!!!!!!")
#     return DistilledRegressionTransformer(base_model, **model_kwargs)


@register_model
def Swin_ViCCT_small(init_path=None, pretrained_cc=False, **kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k
    """

    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)

    base_model = _create_swin_transformer('swin_small_patch4_window7_224', **model_kwargs)

    if init_path and not pretrained_cc:
        base_model = init_model_state(base_model, init_path)

    full_model = DistilledRegressionTransformer(base_model, **model_kwargs)

    if init_path and pretrained_cc:
        full_model = load_pretrained(full_model, init_path)

    full_model.crop_size = 224

    return full_model


@register_model
def Swin_ViCCT_base(init_path=None, pretrained_cc=False, **kwargs):
    """ Swin-B @ 224x224, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18), num_heads=(4, 8, 16), **kwargs)

    # patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)

    base_model = _create_swin_transformer('swin_base_patch4_window7_224', **model_kwargs)

    if init_path and not pretrained_cc:
        base_model = init_model_state(base_model, init_path)

    full_model = DistilledRegressionTransformer(base_model, **model_kwargs)

    if init_path and pretrained_cc:
        full_model = load_pretrained(full_model, init_path)

    full_model.crop_size = 224

    return full_model


@register_model
def Swin_ViCCT_large(init_path=None, pretrained_cc=False, **kwargs):
    """ Swin-L @ 224x224, trained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18), num_heads=(6, 12, 24), **kwargs)
    base_model = _create_swin_transformer('swin_large_patch4_window7_224', **model_kwargs)

    if init_path and not pretrained_cc:
        base_model = init_model_state(base_model, init_path)

    full_model = DistilledRegressionTransformer(base_model, **model_kwargs)

    if init_path and pretrained_cc:
        full_model = load_pretrained(full_model, init_path)

    full_model.crop_size = 224

    return full_model


@register_model
def Swin_ViCCT_large_22k(init_path=None, pretrained_cc=False, **kwargs):
    """ Swin-L @ 224x224, trained ImageNet-22k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18), num_heads=(6, 12, 24), **kwargs)
    base_model = _create_swin_transformer('swin_large_patch4_window7_224_in22k', **model_kwargs)

    if init_path and not pretrained_cc:
        base_model = init_model_state(base_model, init_path)

    full_model = DistilledRegressionTransformer(base_model, **model_kwargs)

    if init_path and pretrained_cc:
        full_model = load_pretrained(full_model, init_path)

    full_model.crop_size = 224

    return full_model