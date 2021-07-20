from typing import Tuple, List, Sequence, Union

import numpy as np
import torch as th
from torch import nn
from torch.nn.utils import spectral_norm

from ail.common.type_alias import Activation, StrToActivation 
from ail.console.color_console import Console


def build_mlp(
    sizes: Sequence[int],
    activation: Activation = "relu",
    output_activation: Activation = nn.Identity(),
    use_spectral_norm=False,
) -> nn.Module:
    """
    Build a feedforward neural network.
    given sizes of each layer.
    
    :param sizes: Sizes of hidden layers.
    :param activation: Activation function.
    :param output_activation: Output Activation function.
        Default nn.Identity().
    :param use_spectral_norm: Apply spectral norm.
        Useful when training a GAN.
    :returns: Neural Network with fully-connected linear layers and
        activation layers.
    """
    # String name to Activation function conversion
    if isinstance(activation, str):
        activation = StrToActivation[activation.lower()].value
    if isinstance(output_activation, str):
        output_activation = StrToActivation[output_activation.lower()].value

    layers = []
    for j in range(len(sizes) - 1):
        activation_fn = activation if j < (len(sizes) - 2) else output_activation
        if use_spectral_norm:
            layers += [spectral_norm(nn.Linear(sizes[j], sizes[j + 1])), activation_fn]
        else:
            layers += [nn.Linear(sizes[j], sizes[j + 1]), activation_fn]
    return nn.Sequential(*layers)


def count_vars(module) -> int:
    """Count number of parameters in neural network."""
    return sum([np.prod(p.shape) for p in module.parameters()])


def orthogonal_init(module: nn.Module, gain: float = 1) -> None:
    """Orthogonal initialization. (used in PPO and A2C)"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def disable_gradient(net: nn.Module) -> None:
    """Freeze the gradient in network."""
    for param in net.parameters():
        param.requires_grad = False


def enable_gradient(net: nn.Module) -> None:
    """Enable gradient in network."""
    for param in net.parameters():
        param.requires_grad = True


def init_gpu(use_gpu=True, gpu_id=0) -> th.device:
    """Initialize device. ('cuda:0' or 'cpu')"""
    if th.cuda.is_available() and use_gpu:
        device = th.device("cuda:" + str(gpu_id))
        Console.info(f"Using GPU id {gpu_id}.\n")
    else:
        device = th.device("cpu")
        if not th.cuda.is_available():
            Console.warn("GPU not detected. Defaulting to CPU.\n")
        elif not use_gpu:
            Console.info("Device: set to use CPU.\n")
    return device


def to_torch(
    array: Union[np.ndarray, Tuple, List],
    device: Union[th.device, str],
    copy: bool = True,
) -> th.Tensor:
    """
    Convert a numpy array to a PyTorch tensor.
    Note: it copies the data by default.
    
    :param array:
    :param device: PyTorch device to which the values will be converted.
    :param copy: Whether to copy or not the data.
        (may be useful to avoid changing things be reference)
    :return: torch tensor.
    """
    if copy:
        return th.tensor(array, dtype=th.float32).to(device)
    elif isinstance(array, np.ndarray):
        return th.from_numpy(array).float().to(device)
    else:
        return th.as_tensor(array, dtype=th.float32).to(device)


def to_numpy(tensor: th.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array and send to CPU."""
    return tensor.detach().cpu().numpy()


def asarray_shape2d(x:Union[th.Tensor, np.ndarray, float, int]) -> np.ndarray:
    """Convert input into numpy array and reshape so that n_dim = 2."""
    if isinstance(x, th.Tensor):
        return to_numpy(x).reshape(1, -1)
    else:
        return np.asarray(x, dtype=np.float32).reshape(1, -1)
