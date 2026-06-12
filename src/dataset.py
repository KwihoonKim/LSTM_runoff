# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:17:38 2026

@author: Kwihoon Kim
"""

import torch
import numpy as np

from torch.utils.data import Dataset

class HydroDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]
