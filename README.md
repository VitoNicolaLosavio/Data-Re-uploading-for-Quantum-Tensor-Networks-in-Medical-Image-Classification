# Data Re-uploading for Quantum Tensor Networks in Medical Image Classification

This repository contains the code and experimental pipeline used for the paper:

**Data Re-uploading for Quantum Tensor Networks in Medical Image Classification**

accepted at **QAI 2026**.

The project investigates quantum feature-map design for multiclass medical image classification 
through a hybrid pipeline combining BiomedCLIP embeddings, a Triplet Autoencoder, 
and an MPS-PBC/bTTN variational quantum classifier. The study compares standard angle encoding, 
H-RY encoding, standard data re-uploading, and a partial re-uploading strategy 
under a fixed four-qubit register.

## Pipeline

The experimental pipeline consists of:

1. Loading the official MedMNIST train/test partitions.
2. Building balanced low-data training subsets.
3. Extracting visual embeddings using BiomedCLIP.
4. Structuring the latent space with a Triplet Autoencoder.
5. Mapping latent features to the interval $[0,\pi]$.
6. Encoding the latent representation using one of four feature-map strategies:
   - `ry`
   - `h-ry`
   - `standard re-uploading`
   - `partial re-uploading`
7. Training a One-vs-All ensemble of MPS-PBC/bTTN quantum classifiers.
8. Evaluating the models using macro-averaged F1-score.

## Dataset

Experiments are performed on **PathMNIST**, distributed as part of MedMNIST.

Only the official training split is subsampled. The official test split remains separate and is used exclusively for final evaluation.

The experiments consider:

- Number of classes: `4, 6, 8, 9`
- Training samples per class: `50, 100, 200, 300`
- Number of qubits: `4`

## Feature Maps

The following encoding strategies are compared:

- **RY angle encoding**
- **H-RY encoding**
- **Standard re-uploading**
- **Partial re-uploading**

For partial re-uploading, a six-dimensional latent representation is sequentially 
injected over adjacent qubit pairs while keeping the quantum register fixed.

## Main Experimental Settings

- BiomedCLIP embeddings: `256` dimensions
- Triplet margin: `0.2`
- Reconstruction weight: `1.0`
- T-AE optimiser: Adam
- Learning rate: `1e-3`
- Quantum optimiser: COBYLA
- Maximum COBYLA iterations: `1000`
- Shots: `4096`
- Random seed: `12345`
- Backend: Qiskit Aer

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Results

Experimental results are automatically stored as CSV files in the results/ directory.

Existing configurations are updated when re-evaluated, while new configurations are appended.

## Reproducibility

Random seeds are fixed across NumPy, TensorFlow, and PyTorch.

## Citation
Citation information will be added after publication.

## License
Please refer to the LICENSE file for usage terms.