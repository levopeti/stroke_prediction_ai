from torch.nn import Module

from imputation.unet import UNet


def define_unet_model(input_shape: int, **kwargs) -> Module:
    return UNet(n_channels=input_shape)


imputation_model_dict = {
    "unet": define_unet_model
}

