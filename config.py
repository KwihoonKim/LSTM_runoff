# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:19:51 2026

@author: Kwihoon Kim
"""

import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEQ_LEN = 168

INPUT_SIZE = 6

BATCH_SIZE = 1024

HIDDEN_SIZE = 64

NUM_LAYERS = 4

LR = 1e-4

EPOCHS = 1500

TAU = 0.5

BASIN_PG = [10000, 35, 1240, 55]
BASIN_GA = [5000, 1580, 390, 90]