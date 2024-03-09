import torch
import torch.nn as nn

import tps.utils.utils as utils


class ModelModule(nn.Module, utils.HyperParameters):
  def __init__(self, plot_train_per_epoch=10, plot_valid_per_epoch=10):
    super().__init__()
    self.save_hyperparameters()

  def loss(self, y_hat, y):
    raise NotImplementedError

  def forward(self, inputs, targets):
    return self(inputs, targets)

  def configure_optimizers(self):
    """Defined in :numref:`sec_classification`"""
    return torch.optim.Adam(self.parameters(), lr=self.lr)
