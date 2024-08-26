# VGAEMalGAN

# VGAE-MalGAN: IoT-Based Android Malware Detection Using Graph Neural Network with Adversarial Defense

## Overview

This repository contains the implementation of the VGAE-MalGAN framework, as introduced in the paper "IoT-Based Android Malware Detection Using Graph Neural Network with Adversarial Defense" published in the IEEE Internet of Things Journal. The project focuses on enhancing Android malware detection using a combination of Graph Neural Networks (GNNs) and adversarial defense mechanisms.

## Abstract

As the Internet of Things (IoT) increasingly relies on Android applications, the detection of malicious Android apps has become critical. This project explores graph-based deep learning approaches for extracting relationships from Android applications in the form of API graphs, which are then used to generate graph embeddings. These embeddings, combined with 'Permission' and 'Intent' features, are employed to train multiple machine learning and deep learning algorithms for malware detection.

The classification process achieved an accuracy of 98.33% on the CICMaldroid dataset and 98.68% on the Drebin dataset. However, the vulnerability of graph-based deep learning models is highlighted by the ability of attackers to introduce fake relationships in API graphs to evade detection. To address this, we propose VGAE-MalGAN, a Generative Adversarial Network (GAN)-based algorithm designed to attack GNN-based Android malware classifiers. VGAE-MalGAN generates adversarial API graphs that deceive the classifier, reducing its detection accuracy. Experimental analysis shows that while the GNN classifier initially fails to detect these adversarial samples, retraining the model with the generated adversarial samples enhances its robustness against such attacks.

[View the full architecture diagram][System Architecture](yumlu6.pdf)
## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.7 or later
- Pytorch
- Pytroch Geometric

### Running Notebooks


1. Clone the repository:
   ```bash
   git clone https://github.com/Yumlembam/VGAEMalGAN.git
   ```

2. Run the following steps to execute the project:

   1. Run `run.py` with the `test-project` argument to extract the API graph:
      ```bash
      python run.py test-project
      ```
   
   2. Run `Graph Neural Network.ipynb` located in the `GNN/` folder to extract the graph embeddings.

   3. Run the machine learning algorithms inside the `Machine Learning/` folder to get the machine learning results:
      - `Decission Tree.ipynb`
      - `Naive Bayes.ipynb`
      - `RandomForrest.ipynb`
      - `SVM.ipynb`
      - `convolutional-training.ipynb`

   4. Run the `VGAEMalGAN` notebooks to generate adversarial API graphs:
      - **Option A**: Run `GAN-VAE.ipynb` when GraphSAGE is used as the black-box model.
      - **Option B**: Run `GAN-VAE-CNN.ipynb` when CNN + GraphSAGE is used as the black-box model.
```


