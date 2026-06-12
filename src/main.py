# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:12:49 2026

@author: Kwihoon Kim
"""

from src.model import LSTM_ALD
from src.train import train_one_epoch
from src.evaluate import evaluate
from src.config import *

model = LSTM_ALD(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
).to(device)

for epoch in range(EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device,
        TAU
    )

    val_loss = evaluate(
        model,
        test_loader,
        device,
        TAU
    )

    print(
        f"Epoch {epoch+1} "
        f"| Train {train_loss:.3f} "
        f"| Val {val_loss:.3f}"
    )
