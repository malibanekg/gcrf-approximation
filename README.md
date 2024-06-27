# Further Results on Structured Regression for Multi-Scale Networks

## Authors
- Milan Bašić (Faculty of Sciences and Mathematics, University of Nǐs)
- Branko Arsić (Faculty of Science, University of Kragujevac)
- Zoran Obradović (Department of Computer and Information Sciences, Temple University)

## Abstract
Gaussian Conditional Random Fields (GCRF) are structured regression models that achieve higher regression accuracy by considering the similarities between objects and outputs of unstructured predictors. However, GCRF models do not scale well with large networks. This paper introduces new estimations for Laplacian eigenvalues and eigenvectors to improve computational efficiency while maintaining high prediction accuracy. The proposed models achieve computational complexity improvements and are validated on three random network types, consistently outperforming previous structured models in accuracy.

## Introduction
This repository contains the implementation of the models discussed in the paper "Further Results on Structured Regression for Multi-Scale Networks." The GCRF model has been enhanced to handle large networks more efficiently by performing calculations on factor graphs rather than on the full graph, using the Kronecker product of graphs for decomposition.

## Key Contributions
1. **Enhanced GCRF Model:** Utilizes new estimations for Laplacian eigenvalues and eigenvectors to improve computational efficiency.
2. **Kronecker Graph Product:** Applies the Kronecker product for graph decomposition, addressing the challenge of Laplacian spectrum characterization.
3. **Model Validation:** Demonstrates high prediction accuracy and computational efficiency on various random network types (random, scale-free, and small-world networks).

## Repository Structure
- `Data_generation/`: Contains the source code for the enhanced GCRF model.
- `GCRF_MSN - approx/`: 
- `GCRF_MSN - approx2/`:
- `GCRF_MSN - baseline/`:
- `GCRF_MSN - proper_Jesse/`: 
- `README.md`: This file.

## Installation
To run the code, you need to have Matlab installed.

## Usage
### Running the Model
To run the models on the provided datasets, use the following command:

```bash
run(runExperiments.m)
```

### Configuration
The configuration file (`configs/config.yaml`) contains various parameters for running the model, including dataset paths, model parameters, and experiment settings.

## Examples
Jupyter notebooks in the `notebooks/` directory provide examples of how to use the code. You can run these notebooks to understand the model's functionality and see the results on sample datasets.

## Results
The `results/` folder contains the outputs of the experiments, including prediction accuracy and computational performance metrics.


## Contact
For any questions or issues, please contact:
- Milan Bašić: basic.milan@yahoo.com
- Branko Arsić: brankoarsic@kg.ac.rs
- Zoran Obradović: zoran.obradovic@temple.edu
