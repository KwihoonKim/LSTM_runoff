# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:17:02 2026

@author: Kwihoon Kim
"""

import torch.nn as nn

class LSTM_ALD(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=4,
        dropout=0.2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(
            hidden_size,
            2
        )

        self.softplus = nn.Softplus()

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        mu = out[:, 0]

        b = self.softplus(
            out[:, 1]
        ) + 1e-6

        return mu, b