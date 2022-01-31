import numpy as np
import matlab
import torch
import torch.nn as nn


def unfold(x, mode):
    return np.reshape(np.moveaxis(x, mode, 0), (x.shape[mode], -1))


def estimate_ranks(weights, energy_factor):
    ranks = []

    for mode in range(2):
        unfolded = unfold(weights, mode)
        _, S, _ = np.linalg.svd(unfolded, full_matrices=False)

        total_sum = np.sum(S ** 2)
        count = 0

        for i in S:
            count += 1
            energy = i ** 2 / total_sum
            if energy > energy_factor:
                ranks.append(count)
                break

    return ranks


def mat_tucker_decomposition(layer, eng, energy_factor):
    is_bias = torch.is_tensor(layer.bias)
    weights = layer.weight.cpu().data.numpy()
    R1, R2 = estimate_ranks(weights, energy_factor)
    matlab_weights = matlab.double(weights.tolist())
    factors = eng.conv_tucker2(matlab_weights, R1, R2)

    last = np.asarray(factors[0], dtype=np.float32)
    first = np.asarray(factors[1], dtype=np.float32)
    core = np.asarray(factors[2], dtype=np.float32)

    print(weights.shape)
    print(first.shape)
    print(np.expand_dims(np.moveaxis(first, 0, 1), axis=(2, 3)).shape)
    print(core.shape)
    print(last.shape)
    print(np.expand_dims(last, axis=(2, 3)).shape)

    first_weights = np.expand_dims(np.moveaxis(first, 0, 1), axis=(2, 3)).copy()
    last_weights = np.expand_dims(last, axis=(2, 3)).copy()

    first_layer = torch.nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    core_layer = torch.nn.Conv2d(
        in_channels=core.shape[1],
        out_channels=core.shape[0],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    last_layer = torch.nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
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
