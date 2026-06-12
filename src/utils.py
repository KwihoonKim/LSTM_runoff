# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:14:16 2026

@author: Kwihoon Kim
"""

import numpy as np
import torch

def prediction_and_actual(
    model,
    loader,
    device
):

    predictions = []
    actuals = []

    model.eval()

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            mu, _ = model(x)

            predictions.append(
                mu.cpu().numpy()
            )

            actuals.append(
                y.numpy()
            )

    return (
        np.concatenate(predictions),
        np.concatenate(actuals)
    )
