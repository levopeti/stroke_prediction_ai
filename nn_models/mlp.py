import torch
from torch.nn import Module, Linear, ReLU, ModuleList


class MLP(Module):
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 layer_sizes: list):
        super().__init__()
        assert len(layer_sizes) > 0, layer_sizes
        self.layers = ModuleList([Linear(input_shape, layer_sizes[0])])

        for i in range(len(layer_sizes) - 1):
            self.layers.append(ReLU())
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.layers.append(Linear(layer_sizes[-1], output_shape))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.squeeze(x)
        return x
