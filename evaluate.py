# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:14:59 2026

@author: Kwihoon Kim
"""

from loss import ald_loss
import torch

def evaluate(
    model,
    loader,
    device,
    tau
):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            mu, b = model(x)

            loss = ald_loss(
                mu,
                b,
                y,
                tau
            )

            total_loss += loss.item()

    return total_loss