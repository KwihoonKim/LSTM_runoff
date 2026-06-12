# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:18:19 2026

@author: Kwihoon Kim
"""

import numpy as np
import pandas as pd

def calculate_effective_rainfall(df, cn):

    threshold = 0.0

    df["is_rain"] = df[1] > threshold

    df["event_id"] = (
        df["is_rain"] != df["is_rain"].shift()
    ).cumsum()

    df.loc[~df["is_rain"], "event_id"] = np.nan

    df["P_cum"] = (
        df.groupby("event_id")[1]
        .cumsum()
    )

    S = 25400 / cn - 254

    P = df["P_cum"]

    Pe_cum = np.where(
        P > 0.2 * S,
        ((P - 0.2*S)**2)/(P + 0.8*S),
        0
    )

    Pe_cum = pd.Series(
        Pe_cum,
        index=df.index
    )

    df["effective"] = (
        Pe_cum.groupby(df["event_id"])
        .diff()
        .fillna(Pe_cum)
    )

    return df