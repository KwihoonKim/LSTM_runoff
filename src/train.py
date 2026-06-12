# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:16:17 2026

@author: Kwihoon Kim
"""

from loss import ald_loss

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    tau
):

    model.train()

    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        mu, b = model(x)

        loss = ald_loss(
            mu,
            b,
            y,
            tau
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss
