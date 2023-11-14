# Comparative Analysis of CNN Architectures on CIFAR-10 Dataset

## Overview
This project evaluates two distinct Convolutional Neural Network (CNN) architectures: SimCNN and ResNet50. The goal is to benchmark their performance on the CIFAR-10 dataset through a series of experiments that involve training the models, tuning hyperparameters, and comparing their test accuracies.

## Model Architectures

### SimCNN Architecture
- Sequential model with:
  - Convolutional layers with 32 and 64 filters of size 3x3.
  - ReLU activation functions.
  - Batch normalization.
  - Max pooling layer of size 2x2.
  - Dense layers with 64 neurons (ReLU) and 10 neurons (SoftMax).

### ResNet50 Architecture
- Pre-trained on ImageNet and modified for CIFAR-10 with:
  - GlobalAveragePooling2D layer.
  - Dense layers with 256 neurons (ReLU) and 10 neurons (SoftMax).

## Parameters
- Batch Size: 32
- Steps per Epoch: 550
- Epochs: 25
- Validation Steps: 1
- Optimizer: SGD

## Training Results
- SimCNN reaches 93.49% training accuracy.
- ResNet50 achieves 95.49% training accuracy.
- Training time approximately 11 hours.

## Test Accuracy
- SimCNN: 68.26%
- ResNet50: 80.48%

## Hyperparameter Tuning for ResNet50
- Increased Batch Size to 64.
- Increased Steps per Epoch to 1000.
- Reduced Epochs to 10.
- Increased Validation Steps to 2.
- Changed Optimizer to ADAM.

## Results Post Hyperparameter Tuning
- Training Accuracy: Starts at 59%, peaks at 86.93%.
- Validation Accuracy: Starts at 62%, peaks at 81.25%.
- Test Accuracy: 74%.

## Conclusions
- The ResNet50 model outperforms SimCNN on the CIFAR-10 dataset.
- Hyperparameter tuning shows a potential increase in validation accuracy but a decrease in test accuracy.

## Recommendations
- Further experiments with grid search for hyperparameter optimization.
- Consideration of dataset and generator capabilities in relation to steps_per_epoch and epochs for training efficiency.

