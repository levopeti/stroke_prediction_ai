from torch.nn import Module

from nn_models.inception_time import InceptionTime
from nn_models.mlp import MLP


def define_mlp_model(input_shape: int, output_shape: int, layer_sizes: list, **kwargs) -> Module:
    return MLP(input_shape, output_shape, layer_sizes)


def define_inception_time(input_shape: int, output_shape: int, **kwargs) -> Module:
    return InceptionTime(in_channels=input_shape, out_size=output_shape)


model_dict = {
    "mlp": define_mlp_model,
    "inception_time": define_inception_time,
}
