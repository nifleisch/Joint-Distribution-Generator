from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import Tensor


def generate_random_corr_matrix(d: int) -> Tensor:
    A = torch.rand(d, d)
    corr_matrix = A @ A.t()
    D = torch.diag(1.0 / torch.sqrt(torch.diag(corr_matrix)))
    corr_matrix = D @ corr_matrix @ D
    return corr_matrix


def plot_marginal_distributions(
    distributions: List,
    sample: Tensor,
    filepath: Path) -> None:
    d = len(distributions)
    rows = (d + 2) // 3
    fig, axs = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axs = axs.flatten()
    for i in range(d):
        marginal_sample = sample[:, i].numpy()
        if is_discrete(distributions[i]):
            values, counts = np.unique(marginal_sample, return_counts=True)
            counts = counts / counts.sum()
            axs[i].bar(values, counts, alpha=0.6, color='b')
            pmf_values = distributions[i].pmf(values)
            axs[i].plot(values, pmf_values, 'rx', lw=2, label='True PMF')
        else:
            axs[i].hist(marginal_sample, bins=100, density=True, edgecolor='black', alpha=0.6)
            x = np.linspace(marginal_sample.min(), marginal_sample.max(), 1000)
            pdf_values = distributions[i].pdf(x)
            axs[i].plot(x, pdf_values, 'r', lw=2, label='True PDF')
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('Frequency')
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def plot_corr_matrices(
    corr_matrix_target: Tensor,
    corr_matrix_sample: Tensor,
    filepath: Path):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    sns.heatmap(corr_matrix_target.numpy(), annot=True, fmt=".2f", cmap='coolwarm', ax=axs[0])
    axs[0].set_title('Target Correlation Matrix')

    sns.heatmap(corr_matrix_sample.numpy(), annot=True, fmt=".2f", cmap='coolwarm', ax=axs[1])
    axs[1].set_title('Sample Correlation Matrix')

    plt.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def is_discrete(dist):
    return hasattr(dist, 'pmf')
