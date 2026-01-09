# CFF

This repository provides **simple example scripts** illustrating how to extract
Deeply Virtual Compton Scattering (DVCS) **Compton Form Factors (CFFs)** using
modern inference techniques based on **deep neural networks**.

The examples are intended to demonstrate practical implementations of the
methods discussed in the accompanying methodology note:

> **Exploring DNN Extraction of DVCS CFFs**  

## Scope

- Demonstrates *how* CFF extraction pipelines can be constructed  
- Focuses on inference strategy rather than producing a finalized global fit  
- Designed as a reference and starting point for further development  

This code is **not** a turnkey analysis framework, but rather a compact set of
worked examples meant to clarify:
- local vs. global (KMI-style) extraction ideas,
- uncertainty propagation with replicas / ensembles,
- separation of physics modeling from neural-network learning.

## Intended audience

Researchers familiar with DVCS phenomenology and CFF extraction who want
concrete, minimal examples of how modern ML-based inference techniques can be
implemented in practice.

You will need TensorFlow, and the BMK library that is compatible with TensorFlow
which you can get by doing
```bash
pip install bmk10
```
If want to use the Gepard you can look here: https://github.com/kkumer/gepard

## Status

Active development.  
Interfaces, scripts, and examples may evolve as the methodology is refined.
