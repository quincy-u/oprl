import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        elif len(x.shape) == 3:
            if x.shape[0] == self.num_ensemble:
                x = torch.einsum('bij,bjk->bik', x, weight)
            else:
                x = torch.einsum('cij,bjk->bcik', x, weight) # this is of shape (n_ensemble, batch_size, segment_length, dim)
        else:
            # 4D case, (n_ensembles, batch_size, segment_length, dim) -> (n_ensemble, batch_size, segment_length, next_dim)
            x = torch.einsum('bcij,bjk->bcik', x, weight)

        if len(x.shape) == 3:
            # (n_ensemble, batch_size, dim)
            x = x + bias
        else:
            # (num_ensemble, batch_size, segment_length, dim) -> unsqueeze to segment length because we want the same bias across the segment length
            bias = bias.unsqueeze(2) # (n_ensemble, 1, 1, output_dim) -- autotiling will take place on the 1st and 2nd dimension I think
            x = x + bias

        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss