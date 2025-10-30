# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

class AsyncMatrix(nn.Module):
    '''
    An asynchronous matrix
    @param generation_len: int - generation len
    @param n_steps: int - number of FM steps to adjust async time scheduler accordingly 
    '''
    def __init__(
        self, 
        generation_len : int, 
        n_steps : int | None = None
    ):
        super().__init__()

        if n_steps is None:
            n_steps = generation_len * 2
        assert n_steps >= generation_len

        dt = 1. / n_steps
        self.time_step = dt
        self.n_steps = n_steps

        self.end_times = nn.Parameter(
            torch.flip( torch.arange(0, generation_len, dtype=torch.float32) * dt, (0,) ) + 1.,
            requires_grad=False
        ) # (L,)
        
        self.multiplier = nn.Parameter(
            self.end_times[0].clone().detach(),
            requires_grad=False
        )

        self.times = nn.Parameter(
            torch.flip(torch.linspace(0., 1., n_steps + 1),(0,)), 
            requires_grad=False
        )
    
    def _get_raw_t(self, t: torch.Tensor):
        '''
        t: (bs, 1)
        '''
        assert len(t.shape) == 2
        assert t.size(1) == 1
        return (self.end_times.unsqueeze(0) - t * self.multiplier)

    def forward(self, t: torch.Tensor):
        '''
        t: (bs, 1)
        :Output:
        A_t: (bs, L)
        '''
        raw_t = self._get_raw_t(t)
        return torch.clamp(raw_t, 0, 1)

    def derivative(self, t: torch.Tensor):
        '''
        t: (bs, 1)
        :Output:
        A_t': (bs, L)
        '''
        raw_t = self._get_raw_t(t)
        is_valid = (raw_t < 1.) & (raw_t > 0.)
        return -1 * self.multiplier * is_valid

    def __call__(self, t):
        return self.forward(t)


class DisjointMatrix(nn.Module):
    '''
    A disjoint matrix
    @param num_rows: torch.tensor - number of rows for each table in batch. Shape: (batch_size)
    @param max_rows: int/long - maximum number of rows.
    '''
    def __init__(self, num_rows, max_rows):
        super().__init__()
        self.max_rows = max_rows
        self.num_rows = num_rows
        self.time_step = 1 / (max_rows + max_rows - 1)

        batch_size = num_rows.shape[0]

        self.times = nn.Parameter(torch.flip(torch.linspace(0,1,1+max_rows),(0,)), requires_grad=False)

        self.start_times = nn.Parameter(self.times[1:].repeat(batch_size,1), requires_grad=False)
        self.end_times = nn.Parameter(self.times[:-1].repeat(batch_size,1), requires_grad=False)

        col_indices = torch.arange(max_rows).unsqueeze(0).expand(batch_size, -1)
        mask = col_indices >= num_rows.unsqueeze(1)

        # Set the values to zero where the mask is True
        self.start_times[mask] = -1
        self.end_times[mask] = 0

        # Set mask for attention
        self.attn_mask = torch.ones((batch_size, 1, max_rows, max_rows))
        self.attn_mask[mask.unsqueeze(1).expand(-1, max_rows, -1)[:,None,:,:]] = 0
        self.denom = nn.Parameter(1 / (self.end_times - self.start_times), requires_grad=False)

    def forward(self, t):
        return torch.clamp((self.end_times - t) * self.denom, 0, 1)

    def derivative(self, t):
        is_valid = (t < self.end_times) & (t > self.start_times)
        return -1 * self.denom * is_valid

    def __call__(self, t):
        return self.forward(t)


class SyncMatrix(nn.Module):
    '''
    An synchronous matrix
    @param num_rows: torch.tensor - number of rows for each table in batch. Shape: (batch_size)
    @param max_rows: int/long - maximum number of rows.
    '''
    def __init__(self, num_rows, max_rows):
        super().__init__()
        self.max_rows = max_rows
        self.num_rows = num_rows
        self.time_step = 1 / (max_rows + max_rows - 1)

        batch_size = num_rows.shape[0]

        self.times = nn.Parameter(torch.flip(torch.linspace(0,1,2*max_rows),(0,)), requires_grad=False)

        self.start_times = nn.Parameter(self.times[max_rows:].repeat(batch_size,1), requires_grad=False)
        self.end_times = nn.Parameter(self.times[:max_rows].repeat(batch_size,1), requires_grad=False)

        col_indices = torch.arange(max_rows).unsqueeze(0).expand(batch_size, -1)
        mask = col_indices >= num_rows.unsqueeze(1)

        # Set the values to zero where the mask is True
        self.start_times[mask] = -1
        self.end_times[mask] = 0

        # Set mask for attention
        self.attn_mask = torch.ones((batch_size, 1, max_rows, max_rows))
        self.attn_mask[mask.unsqueeze(1).expand(-1, max_rows, -1)[:,None,:,:]] = 0
        self.denom = nn.Parameter(1 / (self.end_times - self.start_times), requires_grad=False)

    def forward(self, t):
        return (1 - t) * torch.ones_like(self.end_times)

    def derivative(self, t):
        return -torch.ones_like(self.end_times)

    def __call__(self, t):
        return self.forward(t)

def obtain_noise_schedule(As):
    if As == "disjoint":
        return DisjointMatrix
    if As == "sync":
        return SyncMatrix
    if As == "async":
        return AsyncMatrix
    raise Exception(f'Unknown noise schedule {As}')
