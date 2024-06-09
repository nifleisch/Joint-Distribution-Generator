from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
from scipy import stats
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from data import RandomDataset
from model import GenerationNet
from utils import generate_random_corr_matrix, plot_marginal_distributions, plot_corr_matrices


def train(distributions: List, corr_matrix: Tensor) -> None:
    # define hyperparams
    d_latents = 2*d # dimension of latent rv
    n_samples = 2000 # number of samples used for training
    lr = 0.0314
    batch_size = 128
    n_epochs = 20 # len(dataset) = 4096
    accelerator = "cpu" # change to 'gpu' if available

    # the random dataset samples 'n_samples' times 'd_latents'-dimensional uniform noise
    dataset = RandomDataset(n_samples, d_latents)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=7,
                        shuffle=True,
                        persistent_workers=True)

    model = GenerationNet(distributions=distributions,
                          corr_matrix=corr_matrix,
                          d_latents=d_latents,
                          n_samples=n_samples,
                          lr=lr)

    trainer = pl.Trainer(max_epochs=n_epochs,
                         accelerator=accelerator)

    trainer.fit(model, loader)

    sample_size = 10000 # sample size for evaluation
    filename_marginals = 'marginals.png'
    filename_corr = 'correlation.png'

    filepath_marginals = Path("plots", filename_marginals) # takes long for large sample sizes
    filepath_corr = Path("plots", filename_corr)
    sample = dataset.sample(sample_size)
    sample = model.transform_sample(sample)
    corr_matrix_sample = torch.corrcoef(sample.T)

    plot_corr_matrices(corr_matrix_target=corr_matrix,
                       corr_matrix_sample=corr_matrix_sample,
                       filepath=filepath_corr)

    plot_marginal_distributions(distributions=distributions,
                                sample=sample,
                                filepath=filepath_marginals)


if __name__ == "__main__":
    # TODO: define the marginal distributions here
    distributions = [
        stats.norm(loc=-3, scale=1.7),
        stats.uniform(loc=5.3, scale=10.7),
        stats.bernoulli(p=0.314),
        stats.binom(n=10, p=0.42),
        stats.poisson(mu=3),
        stats.expon(scale=1),
        stats.gamma(a=2, scale=1),
        stats.beta(a=2, b=5),
        stats.chi2(df=2),
        stats.t(df=10),
        stats.lognorm(s=0.954, scale=np.exp(0)),
        stats.weibull_min(c=1.5),
    ]
    d = len(distributions)
    # TODO: define the desired correlation matrix (random by default)
    corr_matrix = generate_random_corr_matrix(d)
    train(distributions, corr_matrix)
