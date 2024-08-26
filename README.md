# VGAEMalGAN

# VGAE-MalGAN: IoT-Based Android Malware Detection Using Graph Neural Network with Adversarial Defense

## Overview

This repository contains the implementation of the VGAE-MalGAN framework, as introduced in the paper "IoT-Based Android Malware Detection Using Graph Neural Network with Adversarial Defense" published in the IEEE Internet of Things Journal. The project focuses on enhancing Android malware detection using a combination of Graph Neural Networks (GNNs) and adversarial defense mechanisms.

## Abstract

As the Internet of Things (IoT) increasingly relies on Android applications, the detection of malicious Android apps has become critical. This project explores graph-based deep learning approaches for extracting relationships from Android applications in the form of API graphs, which are then used to generate graph embeddings. These embeddings, combined with 'Permission' and 'Intent' features, are employed to train multiple machine learning and deep learning algorithms for malware detection.

The classification process achieved an accuracy of 98.33% on the CICMaldroid dataset and 98.68% on the Drebin dataset. However, the vulnerability of graph-based deep learning models is highlighted by the ability of attackers to introduce fake relationships in API graphs to evade detection. To address this, we propose VGAE-MalGAN, a Generative Adversarial Network (GAN)-based algorithm designed to attack GNN-based Android malware classifiers. VGAE-MalGAN generates adversarial API graphs that deceive the classifier, reducing its detection accuracy. Experimental analysis shows that while the GNN classifier initially fails to detect these adversarial samples, retraining the model with the generated adversarial samples enhances its robustness against such attacks.

## Repository Structure

- `src/`: Contains the main Python scripts and code for implementing the VGAE-MalGAN framework.
- `notebooks/`: Jupyter notebooks used for experiments, including graph embedding generation, model training, and adversarial attack simulations.
  - `GNN/`: Focuses on Graph Neural Network implementations.
  - `MachineLearning/`: Contains various machine learning model implementations.
  - `VGAEMalGAN/`: Contains notebooks specific to the VGAE-MalGAN model.
- `data/`: Stores the datasets used for training and testing the models.
- `config/`: Configuration files for setting parameters used in the experiments.
- `models/`: Pre-trained models and checkpoints.
- `results/`: Output files, logs, and result metrics from experiments.
- `README.md`: This file, providing an overview and instructions for the repository.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.7 or later
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Yumlembam/VGAEMalGAN.git





1.Run run.py test-project and extract API graph <br/>
2.Run Graph Neural Network.ipynb to extract embedding <br />
3.Run Machinelearning alogrithm inside machine learning folder to get machine learning results <br />
4.Run VGAEMalGAN to generate adverserial API graph a)GAN-VAE.ipynb when graph sage is black box model b) GAN-VAE-CNN when CNN+GraphSage is black box model 
