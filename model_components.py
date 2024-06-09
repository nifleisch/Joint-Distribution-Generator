import torch
from torch import nn


class ClampedActivation(nn.Module):
    """Shift input to range [a, b] using the Sigmoid activation."""
    def __init__(self, a: float, b: float):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return self.a + (self.b - self.a) * torch.sigmoid(x)


class ShiftedReLU(nn.Module):
    """Shift input to range [a, inf] or [-inf, b] using the ReLU activation."""
    def __init__(self, a: float = None, b: float = None):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        if self.a is not None:
            relu_output = torch.relu(x)
            shifted_output = relu_output + self.a
            return shifted_output
        elif self.b is not None:
            relu_output = torch.relu(x)
            shifted_output = torch.clamp(relu_output, max=self.b)
            return shifted_output
        else:
            raise ValueError("One of a or b must be specified.")
