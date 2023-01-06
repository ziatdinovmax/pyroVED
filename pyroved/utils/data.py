from typing import Tuple, Type

import torch


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
    device_ = kwargs.get("device")
    generator_ = torch.Generator(device_) if device_ else None
    batch_size = kwargs.get("batch_size", 100)
    tensor_set = torch.utils.data.dataset.TensorDataset(*args)
    if random_sampler:
        sampler = torch.utils.data.RandomSampler(tensor_set)
        data_loader = torch.utils.data.DataLoader(
            dataset=tensor_set, batch_size=batch_size, sampler=sampler,
            generator=generator_)
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=tensor_set, batch_size=batch_size, shuffle=shuffle,
            generator=generator_)
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
