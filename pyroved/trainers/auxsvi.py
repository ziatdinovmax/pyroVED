from typing import Type, Optional, Union, Dict

from collections import OrderedDict
from copy import deepcopy as dc

import torch
import torch.nn as nn
import pyro
import pyro.infer as infer
import pyro.optim as optim

from ..utils import set_deterministic_mode, average_weights


class auxSVItrainer:
    """
    Stochastic variational inference (SVI) trainer for variational models
    with auxillary losses

    Args:
        model:
            Initialized model. Must be a subclass of torch.nn.Module
            and have self.model and self.guide methods
        optimizer:
            Pyro optimizer (Defaults to Adam with learning rate 5e-4)
        seed:
            Enforces reproducibility

    Keyword Args:
        lr: learning rate (Default: 5e-4)
        device:
            Sets device to which model and data will be moved.
            Defaults to 'cuda:0' if a GPU is available and to CPU otherwise.

    Examples:

    >>> # Initialize model for semi supervised learning
    >>> data_dim = (28, 28)
    >>> ssvae = pyroved.models.ssiVAE(data_dim, latent_dim=2, num_classes=10, invariances=['r'])
    >>> # Initialize SVI trainer for models with auxiliary loss terms
    >>> trainer = auxSVItrainer(ssvae)
    >>> # Train for 200 epochs:
    >>> for _ in range(200):
    >>>     trainer.step(loader_unsuperv, loader_superv, loader_valid)
    >>>     trainer.print_statistics()
    """

    def __init__(self,
                 model: Type[nn.Module],
                 task: str = "classification",
                 optimizer: Type[optim.PyroOptim] = None,
                 seed: int = 1,
                 **kwargs: Union[str, float]
                 ) -> None:
        """
        Initializes trainer parameters
        """
        pyro.clear_param_store()
        set_deterministic_mode(seed)
        if task not in ["classification", "regression"]:
            raise ValueError("Choose between 'classification' and 'regression' tasks")
        self.task = task
        self.device = kwargs.get(
            "device", 'cuda' if torch.cuda.is_available() else 'cpu')
        if optimizer is None:
            lr = kwargs.get("lr", 5e-4)
            optimizer = optim.Adam({"lr": lr})
        if self.task == "classification":
            guide = infer.config_enumerate(
                model.guide, "parallel", expand=True)
            loss = pyro.infer.TraceEnum_ELBO(
                max_plate_nesting=1, strict_enumeration_warning=False)
        else:
            guide = model.guide
            loss = pyro.infer.Trace_ELBO()
        
        self.loss_basic = infer.SVI(
            model.model, guide, optimizer, loss=loss)
        self.loss_aux = infer.SVI(
            model.model_aux, model.guide_aux,
            optimizer, loss=pyro.infer.Trace_ELBO())
        self.model = model

        self.history = {"training_loss": [], "test": []}
        self.current_epoch = 0
        self.running_weights = {}

    def compute_loss(self,
                     xs: torch.Tensor,
                     ys: Optional[torch.Tensor] = None,
                     **kwargs: float) -> float:
        """
        Computes basic and auxillary losses
        """
        xs = xs.to(self.device)
        if ys is not None:
            ys = ys.to(self.device)
        loss = self.loss_basic.step(xs, ys, **kwargs)
        loss_aux = self.loss_aux.step(xs, ys, **kwargs)
        return loss + loss_aux

    def train(self,
              loader_unsup: Type[torch.utils.data.DataLoader],
              loader_sup: Type[torch.utils.data.DataLoader],
              **kwargs: float
              ) -> float:
        """
        Train a single epoch
        """
        # Get info on number of supervised and unsupervised batches
        sup_batches = len(loader_sup)
        unsup_batches = len(loader_unsup)
        p = (sup_batches + unsup_batches) // sup_batches

        loader_sup = iter(loader_sup)
        epoch_loss = 0.
        unsup_count = 0
        for i, (xs,) in enumerate(loader_unsup):
            # Compute and store loss for unsupervised part
            epoch_loss += self.compute_loss(xs, **kwargs)
            unsup_count += xs.shape[0]
            if i % p == 1:
                # sample random batches xs and ys
                xs, ys = next(loader_sup)
                # Compute supervised loss
                _ = self.compute_loss(xs, ys, **kwargs)

        return epoch_loss / unsup_count

    def evaluate(self,
                 loader_val: Optional[torch.utils.data.DataLoader]) -> float:
        """
        Evaluates model's current state on labeled test data
        """
        if self.task == "classification":
            return self.evaluate_cls(loader_val)
        return self.evaluate_reg(loader_val)

    def evaluate_cls(self,
                     loader_val: Optional[torch.utils.data.DataLoader]) -> float:
        correct, total = 0, 0
        with torch.no_grad():
            for data, labels in loader_val:
                predicted = self.model.classifier(data)
                _, lab_idx = torch.max(labels.cpu(), 1)
                correct += (predicted == lab_idx).sum().item()
                total += data.size(0)
        return correct / total

    def evaluate_reg(self,
                     loader_val: Optional[torch.utils.data.DataLoader]) -> float:
        correct, total = 0, 0
        with torch.no_grad():
            for data, gt in loader_val:
                predicted = self.model.regressor(data)
                mse = nn.functional.mse_loss(predicted, gt)
                correct += mse
                total += 1
        return correct / total

    def step(self,
             loader_unsup: torch.utils.data.DataLoader,
             loader_sup: torch.utils.data.DataLoader,
             loader_val: Optional[torch.utils.data.DataLoader] = None,
             **kwargs: float
             ) -> None:
        """
        Single train (and evaluation, if any) step.

        Args:
            loader_unsup:
                Pytorch's dataloader with unlabeled training data
            loader_sup:
                Pytorch's dataloader with labeled training data
            loader_val:
                Pytorch's dataloader with validation data
            **scale_factor:
                Scale factor for KL divergence. See e.g. https://arxiv.org/abs/1804.03599
                Default value is 1 (i.e. no scaling)
            **aux_loss_multiplier:
                Hyperparameter that modulates the importance of the auxiliary loss
                term. See Eq. 9 in https://arxiv.org/abs/1406.5298. Default values is 20.
        """
        train_loss = self.train(loader_unsup, loader_sup, **kwargs)
        self.history["training_loss"].append(train_loss)
        if loader_val is not None:
            eval_acc = self.evaluate(loader_val)
            self.history["test"].append(eval_acc)
        self.current_epoch += 1

    def save_running_weights(self, net: str) -> None:
        """
        Saves the running weights of specified neural net (e.g. "encoder_y")
        Usually meant for a classifier neural network
        """
        net = getattr(self.model, net)
        state_dict_ = OrderedDict()
        for k, v in net.state_dict().items():
            state_dict_[k] = dc(v).cpu()
        self.running_weights[self.current_epoch] = state_dict_

    def average_weights(self,
                        net: str
                        ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Updates the selected neural net with an averaged weights
        """
        net = getattr(self.model, net)
        net.load_state_dict(average_weights(self.running_weights))

    def print_statistics(self) -> None:
        """
        Print training and test (if any) losses for current epoch
        """
        e = self.current_epoch
        if len(self.history["test"]) > 0:
            if self.task == "classification":
                template = 'Epoch: {} Training loss: {:.4f}, Test accuracy: {:.4f}'
            else:
                template = 'Epoch: {} Training loss: {:.4f}, Test MSE: {:.4f}'
            print(template.format(e, self.history["training_loss"][-1],
                                  self.history["test"][-1]))
        else:
            template = 'Epoch: {} Training loss: {:.4f}'
            print(template.format(e, self.history["training_loss"][-1]))
