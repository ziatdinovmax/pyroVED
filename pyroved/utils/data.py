from typing import Tuple, Type

import torch
import torch.tensor as tt
from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToTensor


def init_dataloader(*args: torch.Tensor,
                    random_sampler: bool = False,
                    shuffle: bool = True,
                    **kwargs: int
                    ) -> Type[torch.utils.data.DataLoader]:
    """
    Returns initialized PyTorch dataloader, which is used by pyroVED's trainers.
    The inputs are torch Tensor objects containing training data and (optionally)
    labels.

    Example:

    >>> # Load training data stored as numpy array
    >>> train_data = np.load("my_training_data.npy")
    >>> # Transform numpy array to toech Tensor object
    >>> train_data = torch.from_numpy(train_data).float()
    >>> # Initialize dataloader
    >>> train_loader = init_dataloader(train_data)
    """
    batch_size = kwargs.get("batch_size", 100)
    tensor_set = torch.utils.data.dataset.TensorDataset(*args)
    if random_sampler:
        sampler = torch.utils.data.RandomSampler(tensor_set)
        data_loader = torch.utils.data.DataLoader(
            dataset=tensor_set, batch_size=batch_size, sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=tensor_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def init_ssvae_dataloaders(data_unsup: torch.Tensor,
                           data_sup: Tuple[torch.Tensor],
                           data_val: Tuple[torch.Tensor],
                           **kwargs: int
                           ) -> Tuple[Type[torch.utils.data.DataLoader]]:
    """
    Helper function to initialize dataloader for ss-VAE models
    """
    loader_unsup = init_dataloader(data_unsup, **kwargs)
    loader_sup = init_dataloader(*data_sup, sampler=True, **kwargs)
    loader_val = init_dataloader(*data_val, **kwargs)
    return loader_unsup, loader_sup, loader_val


def get_rotated_mnist(rotation_range: Tuple[int]) -> Tuple[torch.Tensor]:

    mnist_trainset = datasets.MNIST(
        root='./data', train=True, download=True, transform=None)
    imstack_train_r = torch.zeros_like(mnist_trainset.data, dtype=torch.float32)
    labels, angles = [], []
    for i, (im, lbl) in enumerate(mnist_trainset):
        theta = torch.randint(*rotation_range, (1,)).float()
        im = im.rotate(theta.item(), resample=Image.BICUBIC)
        imstack_train_r[i] = ToTensor()(im)
        labels.append(lbl)
        angles.append(torch.deg2rad(theta))
    imstack_train_r /= imstack_train_r.max()
    return imstack_train_r, tt(labels), tt(angles)
