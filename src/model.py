#!/usr/bin/env python 3

import torch
import numpy as np
from typing import Tuple


class CNNModel(torch.nn.Module):

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int,
        num_filters: int=8,
        filter_size: int=3,
        num_hidden_layers: int=4,
        dropout: float=0.1
        ) -> None:
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.filter_size = filter_size
        self.stride = 2
        self.num_filters = num_filters
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout = dropout
        layer_stack = []
        out_shape = np.array(input_shape[1:])
        for i in range(self.num_hidden_layers):
    
            if np.any((out_shape - (filter_size-1) - 1)//self.stride + 1) <= 0:
                print(
                    f'/!\ Warning: out_shape would be <= 0; '
                    f'thus reducing num_hidden_layers to {i}'
                )
                break
                
            elif np.any(out_shape < np.ones(3)*self.filter_size):
                print(
                    f'/!\ Warning: out_shape smaller than filter_size; '
                    f'thus reducing num_hidden_layers to {i}'
                )
                break
            
            else:
                out_shape = (out_shape - (filter_size-1) - 1)//self.stride + 1
                layer_stack.extend(
                    [
                        torch.nn.Conv3d(
                            in_channels=input_shape[0] if i==0 else self.num_filters,
                            out_channels=self.num_filters,
                            kernel_size=self.filter_size,
                            stride=self.stride
                        ),
                        torch.nn.BatchNorm3d(num_features=self.num_filters),
                        torch.nn.ReLU(),
                        torch.nn.Dropout3d(p=self.dropout)
                    ]
                )
        layer_stack.extend(
            [
                torch.nn.Flatten(),
                torch.nn.Linear(
                    in_features=np.prod(out_shape)*self.num_filters,
                    out_features=self.num_classes
                )
            ]
        )
        self.model = torch.nn.Sequential(*layer_stack)

    def forward(
        self,
        inputs: torch.Tensor
        ) -> torch.tensor:  
        return self.model(inputs)