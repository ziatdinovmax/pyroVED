from typing import Type, Optional, Union

import torch
import pyro
import pyro.infer as infer
import pyro.optim as optim

from ..utils import set_deterministic_mode


class SVItrainer:
    """
    Stochastic variational inference (SVI) trainer for
    unsupervised and class-conditioned VED models consisting
    of one encoder and one decoder.

    Args:
        model:
            Initialized model. Must be a subclass of torch.nn.Module
            and have self.model and self.guide methods
        optimizer:
            Pyro optimizer (Defaults to Adam with learning rate 1e-3)
        loss:
            ELBO objective (Defaults to pyro.infer.Trace_ELBO)
        enumerate_parallel:
            Exact discrete enumeration for discrete latent variables
        seed:
            Enforces reproducibility
    
    Keyword Args:
        lr: learning rate (Default: 1e-3)
        device:
            Sets device to which model and data will be moved.
            Defaults to 'cuda:0' if a GPU is available and to CPU otherwise.

    Examples:

    Train a model with SVI trainer using default settings

    >>> # Initialize model
    >>> data_dim = (28, 28)
    >>> trvae = pyroved.models.iVAE(data_dim, latent_dim=2, invariances=['r', 't'])
    >>> # Initialize SVI trainer
    >>> trainer = SVItrainer(trvae)
    >>> # Train for 200 epochs:
    >>> for _ in range(200):
    >>>     trainer.step(train_loader)
    >>>     trainer.print_statistics()

    Train a model with SVI trainer with a "time"-dependent KL scaling factor

    >>> # Initialize model
    >>> data_dim = (28, 28)
    >>> rvae = pyroved.models.iVAE(data_dim, latent_dim=2, invariances=['r'])
    >>> # Initialize SVI trainer
    >>> trainer = SVItrainer(rvae)
    >>> kl_scale = torch.linspace(1, 4, 50) # ramp-up KL scale factor from 1 to 4 during first 50 epochs
    >>> # Train
    >>> for e in range(100):
    >>>     sc = kl_scale[e] if e < len(kl_scale) else kl_scale[-1]
    >>>     trainer.step(train_loader, scale_factor=sc)
    >>>     trainer.print_statistics()
    """
    def __init__(self,
                 model: Type[torch.nn.Module],
                 optimizer: Type[optim.PyroOptim] = None,
                 loss: Type[infer.ELBO] = None,
                 enumerate_parallel: bool = False,
                 seed: int = 1,
                 **kwargs: Union[str, float]
                 ) -> None:
        """
        Initializes the trainer's parameters
        """
        pyro.clear_param_store()
        set_deterministic_mode(seed)
        self.device = kwargs.get(
            "device", 'cuda' if torch.cuda.is_available() else 'cpu')
        if optimizer is None:
            lr = kwargs.get("lr", 1e-3)
            optimizer = optim.Adam({"lr": lr})
        if loss is None:
            if enumerate_parallel:
                loss = infer.TraceEnum_ELBO(
                    max_plate_nesting=1, strict_enumeration_warning=False)
            else:
                loss = infer.Trace_ELBO()
        guide = model.guide
        if enumerate_parallel:
            guide = infer.config_enumerate(guide, "parallel", expand=True)   
        self.svi = infer.SVI(model.model, guide, optimizer, loss=loss)
        self.loss_history = {"training_loss": [], "test_loss": []}
        self.current_epoch = 0

    def train(self,
              train_loader: Type[torch.utils.data.DataLoader],
              **kwargs: float) -> float:
        """
        Trains a single epoch
        """
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch returned by the data loader
        for data in train_loader:
            if len(data) == 1:  # VAE mode
                x = data[0]
                loss = self.svi.step(x.to(self.device), **kwargs)
            else:  # VED or cVAE mode
                x, y = data
                loss = self.svi.step(
                    x.to(self.device), y.to(self.device), **kwargs)
            # do ELBO gradient and accumulate loss
            epoch_loss += loss

        return epoch_loss / len(train_loader.dataset)

    def evaluate(self,
                 test_loader: Type[torch.utils.data.DataLoader],
                 **kwargs: float) -> float:
        """
        Evaluates current models state on a single epoch
        """
        # initialize loss accumulator
        test_loss = 0.
        # compute the loss over the entire test set
        with torch.no_grad():
            for data in test_loader:
                if len(data) == 1:  # VAE mode
                    x = data[0]
                    loss = self.svi.step(x.to(self.device), **kwargs)
                else:  # VED or cVAE mode
                    x, y = data
                    loss = self.svi.step(
                        x.to(self.device), y.to(self.device), **kwargs)
                test_loss += loss

        return test_loss / len(test_loader.dataset)

    def step(self,
             train_loader: Type[torch.utils.data.DataLoader],
             test_loader: Optional[Type[torch.utils.data.DataLoader]] = None,
             **kwargs: float) -> None:
        """
        Single training and (optionally) evaluation step

        Args:
            train_loader:
                Pytorch’s dataloader object with training data
            test_loader:
                (Optional) Pytorch’s dataloader object with test data
        
        Keyword Args:
            scale_factor:
                Scale factor for KL divergence. See e.g. https://arxiv.org/abs/1804.03599
                Default value is 1 (i.e. no scaling)
        """
        train_loss = self.train(train_loader, **kwargs)
        self.loss_history["training_loss"].append(train_loss)
        if test_loader is not None:
            test_loss = self.evaluate(test_loader, **kwargs)
            self.loss_history["test_loss"].append(test_loss)
        self.current_epoch += 1

    def print_statistics(self) -> None:
        """
        Prints training and test (if any) losses for current epoch
        """
        e = self.current_epoch
        if len(self.loss_history["test_loss"]) > 0:
            template = 'Epoch: {} Training loss: {:.4f}, Test loss: {:.4f}'
            print(template.format(e, self.loss_history["training_loss"][-1],
                                  self.loss_history["test_loss"][-1]))
        else:
            template = 'Epoch: {} Training loss: {:.4f}'
            print(template.format(e, self.loss_history["training_loss"][-1]))
