# Further Results on Structured Regression for Multi-Scale Networks

## Authors
- __Milan Bašić__ (_Faculty of Sciences and Mathematics, University of Niš_)
- __Branko Arsić__ (_Faculty of Science, University of Kragujevac_)
- __Zoran Obradović__ (_Department of Computer and Information Sciences, Temple University_)

## Abstract
Gaussian Conditional Random Fields (GCRF) are structured regression models that achieve higher regression accuracy by considering the similarities between objects and outputs of unstructured predictors. However, GCRF models do not scale well with large networks. This paper introduces new estimations for Laplacian eigenvalues and eigenvectors to improve computational efficiency while maintaining high prediction accuracy. The proposed models achieve computational complexity improvements and are validated on three random network types, consistently outperforming previous structured models in accuracy.

## Introduction
This repository contains the implementation of the models discussed in the paper "Further Results on Structured Regression for Multi-Scale Networks." The GCRF model has been enhanced to handle large networks more efficiently by performing calculations on factor graphs rather than on the full graph, using the Kronecker product of graphs for decomposition.

## Key Contributions
1. **Kronecker Graph Product:** Applies the Kronecker product for graph decomposition, addressing the challenge of Laplacian spectrum characterization.
2. **Enhanced GCRF Model:** Utilizes new estimations for Laplacian eigenvalues and eigenvectors to improve computational efficiency, since characterizing a
Laplacian spectrum of the Kronecker product of graphs from its factor graphs spectra has remained an open problem.
3. **Model Validation:** Demonstrates high prediction accuracy and computational efficiency on various random network types (random, scale-free, and small-world networks).

## Repository Structure
- __`Data_generation/`__: Contains the source code for the random graph generation.
- __`GCRF_MSN - baseline/`__: GCRF model implementation where the numerical eigendecompositon is performed (the highest regression accuracy).
- __`GCRF_MSN - proper_Jesse/`__: An approximated GCRF model implementation was done on the basis of paper "J. Glass and Z. Obradovic. Structured regression on multiscale networks. IEEE Intelligent Systems, 32(2):23–30, 2017."
- __`GCRF_MSN - approx/`__: An approximation for the spectrum of Laplacian matrix of Kronecker product of graphs is implemented according to the paper "H. Sayama. Estimation of laplacian spectra of direct and strong product graphs. Discrete Applied Mathematics, 205:160–170, 2016."
- __`GCRF_MSN - approx2/`__: An approximation for the spectrum of Laplacian matrix of Kronecker product of graphs is implemented according to the paper "Bašić, M., Arsić, B., & Obradović, Z. Another estimation of Laplacian spectrum of the Kronecker product of graphs. Information Sciences, 609, 605-625, 2022"
- `README.md`: This file.

## Installation
To run the code, you need to have Matlab installed.

## Usage
### Running the Model
To run the models on the provided datasets, use the following command:

```bash
run(runExperiments.m)
```

__Settings:__

- **lines 50 and 51:** set the numbers of nodes for the selected random graphs.
- **lines 62 to 69:** select the graph types (Erdos-Renyi, Barabasi-Albert, or Watts-Strogatz); commented numbers represent the number of edges in the networks (corresponding edge density levels are 10%, 30%, 50%, 60% and 80%, respectively).

## Examples

% number of nodes for the graphs G and H<br />
_n1 = 50;_<br />
_n2 = 100;_

### Erdos-Renyi networks
% Erdos-Renyi networks with 30% of edge density levels <br />
_S1 = GenRandGraphFixedNumLinksER(n1, 367);_<br />
_S2 = GenRandGraphFixedNumLinksER(n2, 1485);_<br />

### Barabasi-Albert networks
% Barabasi-Albert networks with 30% of edge density levels <br />
_S1 = generate_random_graph(n1, 'ba', 9, -1);_<br />
_S2 = generate_random_graph(n2, 'ba', 18, -1);_<br />

### Watts-Strogatz networks
% Watts-Strogatz networks with 80% of edge density levels <br />
_S1 = generate_random_graph(n1, 'ws', 40, -1);_<br />
_S2 = generate_random_graph(n2, 'ws', 80, -1);_<br />
    
## Results
Please contact the authors of the paper for more details.

## Contact
For any questions or issues, please contact:
- Milan Bašić: basic.milan@yahoo.com
- Branko Arsić: brankoarsic@kg.ac.rs
- Zoran Obradović: zoran.obradovic@temple.edu
