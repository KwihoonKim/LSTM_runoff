# LSTM Runoff Modeling
Runoff simulation for ungauged basin (LSTM)

A PyTorch-based LSTM framework for hourly runoff prediction using meteorological data and watershed characteristics.

This project aims to improve runoff simulation in ungauged or data-scarce watersheds by integrating static basin descriptors with dynamic weather observations.

<_Overview_>

    Runoff prediction in ungauged watersheds remains a challenging task because streamflow observations are unavailable or limited.

    This framework combines:

    - Meteorological forcing data
    - Precipitation
    - Air temperature
    - Watershed descriptors
    - Drainage area
    - Population density
    - Elevation difference
    - Curve Number (CN)

    with a Long Short-Term Memory (LSTM) network to simulate hourly runoff.


<_Model Workflow_>

    Meteorological Data
    (Precipitation, Temperature)
            ↓
    Hydrologic Feature Extraction
            │
            ├─ 24-hour rainfall accumulation
            ├─ 168-hour temperature average
            └─ NRCS-CN effective rainfall
            ↓
    Basin Characteristics
    (Area, Density, Elevation, CN)
            ↓
    LSTM Network
            ↓
    Hourly Runoff Prediction
