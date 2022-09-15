import numpy as np
import matlab
import torch
import torch.nn as nn


def mat_tucker_decomposition(layer, eng, R1, R2):
    is_bias = torch.is_tensor(layer.bias)
    weights = layer.weight.cpu().data.numpy()
    matlab_weights = matlab.double(weights.tolist())
    factors = eng.conv_tucker2(matlab_weights, R1, R2)

    last = np.asarray(factors[0], dtype=np.float32)
    first = np.asarray(factors[1], dtype=np.float32)
    core = np.asarray(factors[2], dtype=np.float32)

    first_weights = np.expand_dims(np.moveaxis(first, 0, 1), axis=(2, 3)).copy()
    last_weights = np.expand_dims(last, axis=(2, 3)).copy()

    first_layer = torch.nn.Conv2d(
        in_channels=layer.in_channels,
        out_channels=R1,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    core_layer = torch.nn.Conv2d(
        in_channels=R1,
        out_channels=R2,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    last_layer = torch.nn.Conv2d(
        in_channels=R2,
        out_channels=layer.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=True if is_bias else False,
    )

    if is_bias:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = torch.from_numpy(first_weights)
    last_layer.weight.data = torch.from_numpy(last_weights)
    core_layer.weight.data = torch.from_numpy(core)

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)
