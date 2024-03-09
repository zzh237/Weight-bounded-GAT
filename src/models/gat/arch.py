import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as pyg_data
import torch_geometric.nn as graph_nn
import torch_geometric.transforms as T

import math
import typing as tp
import itertools

import tps.models.base_model as base_model
import tps.utils.utils as utils

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        ##takes a list of quantiles
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss    

class GraphAttnNet(base_model.ModelModule):
  def __init__(self, config_model: dict) -> None:
    super().__init__()
    config = {
      "hparams": {
        "input_num_dim":7,
        "output_dim": 1,
        "num_hid_layers": 1,
        "num_in_heads": 2,
        "num_neighbors": 0.5,
        "num_labeled_nodes": None,
        "trn_ratio": 0.6,
        "inputs_norm": 1.0,
        "weights_bound1": None,
        "weights_bound2": 1.0,
        "windowed": True,
        "num_out_heads": 1,
        "hidden_size": 32,
        "num_hid_layers": 1,
        "lr": 1e-4,
        "dropout": 0.1,
        },
      "name": "Graph Attention Network",
    }
    self.name = config["name"]
    config["hparams"].update(config_model["hparams"])
    self.save_hyperparameters_dict(config)

    self.trn_mask = None
    self.val_mask = None
    self.tst_mask = None
    self.conv1 = graph_nn.GATConv(self.input_num_dim, self.hidden_size, heads=self.num_in_heads, dropout=1 - self.num_neighbors)

    self.hid_convs = []
    for i in range(self.num_hid_layers):
      self.hid_convs.append(
        (graph_nn.Sequential('x, edge_index', [
          (graph_nn.GATConv(self.hidden_size*self.num_in_heads, self.hidden_size, heads=self.num_in_heads, dropout=1 - self.num_neighbors), 'x, edge_index -> x'),
          nn.ELU(inplace=True),
          (nn.Dropout(p=0.1), 'x -> x')
        ]), 'x, edge_index -> x')
      )
    self.hid_convs = graph_nn.Sequential('x, edge_index', self.hid_convs)
    self.conv2 = graph_nn.GATConv(self.hidden_size*self.num_in_heads, self.output_dim, concat=False,
                          heads=self.num_out_heads, dropout=1 - self.num_neighbors)
    

    self.tst_mode = False
    self.float()
    self.mse_loss = nn.MSELoss(reduction="none")
    self.quantile_loss = QuantileLoss([0.1])

  def create_mask(self, num_nodes, mode="trn"):
    mask = torch.ones(num_nodes)
    trn_ratio = self.trn_ratio
    val_ratio = 0.5 - trn_ratio/2
    tst_ratio = 0.5 - trn_ratio/2
    if self.num_labeled_nodes is not None:
      trn_ratio = self.num_labeled_nodes
    num_trn_samples = int(num_nodes * trn_ratio)
    num_val_samples = int(num_nodes * val_ratio)
    num_tst_samples = num_nodes - num_trn_samples - num_val_samples
    if mode == "trn":
      mask[num_trn_samples:] = 0
    elif mode == "val":
      mask[:num_trn_samples] = 0
      mask[num_trn_samples + num_val_samples:] = 0
    else:
      mask[:num_trn_samples + num_val_samples] = 0
    
    # print(f"{mode}: {torch.count_nonzero(mask)}")
    mask = mask.bool()

    return mask

  def loss(self, preds, truth):
    # loss = self.mse_loss(preds, truth).mean(dim=1)
    loss = self.quantile_loss(preds, truth)
    return loss

  def weight_clipping(self, module, f_bound):
    for name, param in module.named_parameters():
      w = param.data
      bound = f_bound / torch.sqrt(param.shape[0] * param.shape[1])
      w = w.clamp(-bound, bound)

  def forward(self, inputs: dict, targets: dict): # x shape [batch_size, sequence_length, feature_dim]
    x = torch.cat((inputs["X"], inputs["X_market"]), dim=2).float()
    edge_index = inputs["edge_index"]
    num_nodes = x.shape[0]
    x = x.reshape(num_nodes, -1)
    targets["returns"] = F.normalize(targets["returns"], dim=0)
    x = torch.div(x, x.sum(dim=-1, keepdim=True).clamp(min=1.))
    x = x * self.inputs_norm
    x = F.dropout(x, p=self.dropout, training=self.training)

    # weights bound1
    if self.weights_bound1 is not None:
      self.weight_clipping(self.conv1, self.weights_bound1)
      self.weight_clipping(self.hid_convs, self.weights_bound1)
      self.weight_clipping(self.conv2, self.weights_bound1)

    x = self.conv1(x, edge_index)
    x = F.elu(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.hid_convs(x, edge_index)
    pred = self.conv2(x, edge_index)

    if self.training:
      if self.trn_mask is None:
        self.trn_mask = self.create_mask(num_nodes, "trn")
      loss = self.loss(pred[self.trn_mask], targets["returns"][self.trn_mask])
    else:
      if self.tst_mode:
        # print("tst mode mask")
        if self.tst_mask is None:
          self.tst_mask = self.create_mask(num_nodes, "tst")
        loss = self.loss(pred[self.tst_mask], targets["returns"][self.tst_mask])
      else:
        if self.val_mask is None:
          self.tst_mask = self.create_mask(num_nodes, "val")
        loss = self.loss(pred[self.val_mask], targets["returns"][self.val_mask])

    return pred, loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)

  def prepare_batch(self, inputs, targets):
    num_nodes = inputs["X"].shape[0]

    x = torch.cat((inputs["X"], inputs["X_market"]), dim=2).float()
    x = x.reshape(num_nodes, -1)
    TopK = 30
    B = torch.corrcoef(x).float()
    B[torch.isnan(B)] = 0
    topk_values, topk_indices = torch.topk(B, k=TopK, dim=-1)
    B_sparse = torch.zeros(B.shape[0], B.shape[0])
    B_sparse = B_sparse.scatter(dim=1, index=topk_indices, src=topk_values) # set the non-topk positions as zero
    B_sparse = torch.triu(B_sparse)
    edges = torch.nonzero(B_sparse).tolist()

    reversed_edges = [(end, src) for src, end in edges]
    edges = [e for tup in edges for e in tup]
    reversed_edges = [e for tup in reversed_edges for e in tup]
    edge_index = [edges, reversed_edges]

    targets["returns"] = torch.clip(targets["returns"], max=5)
    inputs_batch = inputs
    inputs_batch["edge_index"] = torch.Tensor(edge_index).long()
    targets_batch = {"returns": torch.Tensor(targets["returns"]).float()}

    return inputs_batch, targets_batch