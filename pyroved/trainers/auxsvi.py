from typing import Type, Optional

import torch
import torch.nn as nn
import pyro
import pyro.infer as infer
import pyro.optim as optim

from ..utils import set_deterministic_mode


class auxSVItrainer:
    """
    Stochastic variational inference (SVI) trainer for variational models
    with auxillary losses
    """

    def __init__(self,
                 model: Type[nn.Module],
                 optimizer: Type[optim.PyroOptim] = None,
                 seed: int = 1
                 ) -> None:
        """
        Initializes trainer parameters
        """
        pyro.clear_param_store()
        set_deterministic_mode(seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if optimizer is None:
            optimizer = optim.Adam({"lr": 5e-4})
        guide = infer.config_enumerate(model.guide, "parallel", expand=True)
        self.loss_basic = infer.SVI(
            model.model, guide, optimizer,
            loss=(pyro.infer.TraceEnum_ELBO)(max_plate_nesting=1, strict_enumeration_warning=False))
        self.loss_aux = infer.SVI(
            model.model_classify, model.guide_classify,
            optimizer, loss=pyro.infer.Trace_ELBO())
        self.model = model

        self.history = {"training_loss": [], "test_accuracy": []}
        self.current_epoch = 0


    def compute_loss(self,
                     xs: torch.Tensor,
                     ys: Optional[torch.Tensor] = None) -> float:
        """
        Computes basic and auxillary losses
        """
        xs = xs.to(self.device)
        if ys is not None:
            ys = ys.to(self.device)
        loss = self.loss_basic.step(xs, ys)
        loss_aux = self.loss_aux.step(xs, ys)
        return loss + loss_aux

    def train(self,
              loader_unsup: Type[torch.utils.data.DataLoader],
              loader_sup: Type[torch.utils.data.DataLoader],
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
            epoch_loss += self.compute_loss(xs)
            unsup_count += xs.shape[0]
            if i % p == 1:
                # sample random batches xs and ys
                xs, ys = loader_sup.next()
                # Compute supervised loss
                _ = self.compute_loss(xs, ys)

        return epoch_loss / unsup_count

    def evaluate(self,
                 loader_val: Optional[torch.utils.data.DataLoader]) -> None:
        """
        Evaluates model's current state on labeled test data
        """
        correct, total = 0, 0
        with torch.no_grad():
            for data, labels in loader_val:
                predicted = self.model.classifier(data)
                _, lab_idx = torch.max(labels, 1)
                correct += (predicted == lab_idx).sum().item()
                total += data.size(0)
        return correct / total

    def step(self,
             loader_unsup: torch.utils.data.DataLoader,
             loader_sup: torch.utils.data.DataLoader,
             loader_val: Optional[torch.utils.data.DataLoader] = None
             ) -> None:
        train_loss = self.train(loader_unsup, loader_sup)
        self.history["training_loss"].append(train_loss)
        if loader_val is not None:
            eval_acc = self.evaluate(loader_val)
            self.history["test_accuracy"].append(eval_acc)
        self.current_epoch += 1

    def print_statistics(self) -> None:
        """
        Print training and test (if any) losses for current epoch
        """
        e = self.current_epoch
        if len(self.history["test_accuracy"]) > 0:
            template = 'Epoch: {} Training loss: {:.4f}, Test accuracy: {:.4f}'
            print(template.format(e, self.history["training_loss"][-1],
                                  self.history["test_accuracy"][-1]))
        else:
            template = 'Epoch: {} Training loss: {:.4f}'
            print(template.format(e, self.history["training_loss"][-1]))