# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:16:29 2026

@author: Kwihoon Kim
"""

import torch

def ald_loss(mu, b, y, tau=0.5):

    u = y - mu

    rho = torch.where(
        u >= 0,
        tau*u,
        (tau-1)*u
    )

    return torch.mean(
        torch.log(b) + rho/b
    )
