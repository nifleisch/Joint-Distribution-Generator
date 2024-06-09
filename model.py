import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchsort
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric

from utils import is_discrete
from model_components import ClampedActivation, ShiftedReLU


class GenerationNet(pl.LightningModule):
    def __init__(
        self,
        distributions: List,
        corr_matrix: Tensor,
        d_latents: int,
        n_samples:int,
        lr: float) -> None:
        super().__init__()
        self.distributions = distributions
        self.corr_matrix = corr_matrix
        self.d_latents = d_latents
        self.n_samples = n_samples
        self.d = len(distributions)
        self.lr = lr

        self.marginal_target = self.get_marginal_target()
        self.discret_idx = self.get_discret_idx()
        self.regularization_params = self.get_regularization_params()

        self.net = nn.Sequential(
            nn.Linear(d_latents, 4*d_latents),
            nn.BatchNorm1d(4*d_latents),
            nn.LeakyReLU(),
            nn.Linear(4*d_latents, 16*d_latents),
            nn.BatchNorm1d(16*d_latents),
            nn.LeakyReLU(),
            nn.Linear(16*d_latents, 8*self.d),
            nn.BatchNorm1d(8*self.d),
            nn.LeakyReLU(),
            nn.Linear(8*self.d, self.d),
        )
        self.prediction_heads = self.get_prediction_heads()

        self.loss = MeanMetric()
        self.loss_corr = MeanMetric()
        self.loss_marginal = MeanMetric()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_marginal_target(self) -> Tensor:
        """For each distribution evaluate quantiles at 'n_samples' equidistant points betwen 0 and 1."""
        discretization = np.linspace(0, 1, self.n_samples + 2)[1:-1]
        marginal_targets = []
        for distribution in self.distributions:
            quantiles = torch.from_numpy(distribution.ppf(discretization)).unsqueeze(1)
            marginal_targets.append(quantiles)
        return torch.cat(marginal_targets, dim=1).float()

    def get_discret_idx(self):
        """Get indices of all discrete distributions."""
        discret_idx = []
        for idx, distribution in enumerate(self.distributions):
            if is_discrete(distribution):
                discret_idx.append(idx)
        return discret_idx

    def get_regularization_params(self):
        """
        The soft sort we use for computing the quantiles approximates the sorting.
        For each distribution find the regularizaton param that optimizes the approximation.
        """
        regularization_params = []
        param_grid = [0.01, 0.1, 1, 10, 100, 1000]
        for i in range(self.d):
            quantiles = self.marginal_target[:, i]
            indices = torch.randperm(quantiles.size(0))
            shuffled_quantiles = quantiles[indices].unsqueeze(0)
            sort_diffs = []
            for regularization_strength in param_grid:
                approx_quantiles = torchsort.soft_sort(shuffled_quantiles, regularization_strength=regularization_strength).squeeze(0)
                sort_diffs.append((quantiles - approx_quantiles).abs().sum())
            _, min_index = torch.min(torch.tensor(sort_diffs), 0)
            regularization_params.append(param_grid[min_index])
        return regularization_params

    def get_prediction_heads(self):
        """
        Define tranformation that make sure that generated samples stay
        in the support of their respective distributions.
        """
        prediction_heads = []
        for distribution in self.distributions:
            a, b = distribution.support()
            if math.isinf(a) and math.isinf(b):
                prediction_heads.append(nn.Identity())
            elif not math.isinf(a) and math.isinf(b):
                prediction_heads.append(ShiftedReLU(a=a))
            elif math.isinf(a) and not math.isinf(b):
                prediction_heads.append(ShiftedReLU(b=b))
            else:
                prediction_heads.append(ClampedActivation(a, b))
        return prediction_heads

    def forward(self, x):
        # reshape in order to use BatchNorm in net
        batch_size, n_samples, d_latents = x.shape
        x = torch.reshape(x, (-1, d_latents))
        x_pred = self.net(x)
        x_pred = x_pred.reshape(batch_size, n_samples, self.d)
        for i, prediction_head in enumerate(self.prediction_heads):
            x_pred[:, :, i] = prediction_head(x_pred[:, :, i])
        return x_pred

    def corr_loss(self, x_pred):
        corr_losses = []
        batch_size, _, _ = x_pred.shape
        corr_target = self.corr_matrix.to(self.device)
        for i  in range(batch_size):
            corr_pred = torch.corrcoef(x_pred[i].T)
            loss = F.l1_loss(corr_pred, corr_target)
            corr_losses.append(loss)
        return torch.stack(corr_losses).mean()

    def marginal_loss(self, x_pred):
        batch_size, _, _ = x_pred.shape
        marginal_target = self.marginal_target.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        losses = []
        for i in range(self.d):
            std = self.distributions[i].std()
            reg = self.regularization_params[i]
            marginal_pred = torchsort.soft_sort(x_pred[:,:,i],
                                                regularization_strength=reg)
            loss = F.l1_loss(marginal_pred, marginal_target[:,:,i]) * 1 / std
            losses.append(loss)
        return torch.stack(losses).mean()

    def training_step(self, x, x_idx):
        x_pred = self.forward(x)
        loss_corr = self.corr_loss(x_pred)
        loss_marginal = self.marginal_loss(x_pred)
        loss = loss_corr + loss_marginal

        self.loss(loss)
        self.loss_corr(loss_corr)
        self.loss_marginal(loss_marginal)

        self.log('loss', self.loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('loss_corr', self.loss_corr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('loss_marginal', self.loss_marginal, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @torch.inference_mode()
    def transform_sample(self, x):
        x = self.forward(x.unsqueeze(0)).squeeze(0)
        for i in self.discret_idx:
            x[:, i] = torch.round(x[:, i])
        return x
