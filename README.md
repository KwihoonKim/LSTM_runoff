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
    (Area, Elevation, CN)
            ↓
    LSTM Network
            ↓
    Hourly Runoff Prediction

<_Features_>

    Hydrologic Preprocessing
    - NRCS Curve Number (CN) runoff generation
    - Event-based cumulative rainfall calculation
    - Effective rainfall estimation
    - Antecedent rainfall accumulation
    - Temperature averaging

    Deep Learning
    - Multi-layer LSTM
    - PyTorch implementation
    - GPU acceleration (Not applied in LSTM)
    - Asymmetric Laplace Distribution (ALD) loss
    - Quantile-based prediction framework
    
    Evaluation
    - Nash-Sutcliffe Efficiency (NSE)
    - Training and validation monitoring
    - Multi-basin performance assessment

<_Input Variables_>

    Variable	            Description
    Precipitation	        Hourly rainfall
    Temperature	            Hourly air temperature
    Effective Rainfall	    NRCS-CN derived runoff-producing rainfall
    Rainfall Accumulation	Previous 24-hour rainfall
    Basin Area	            Watershed area

<_Watershed Attributes_>

    Static watershed descriptors are used as additional predictors:

    Attribute	            Purpose
    Area	                Basin scale
    Elevation Difference	Topographic variability
    Curve Number (CN)	    Runoff generation potential

<_Model Architecture_>

    Input Sequence (168 hr)
            ↓
    LSTM Layer (4 layers)
            ↓
    Fully Connected Layer
            │
            ├── μ (runoff estimate)
            └── b (uncertainty scale)

    Loss Function:

    Asymmetric Laplace Distribution (ALD)












