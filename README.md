# Digit Prediction from Scratch

This project implements a neural network from scratch to predict handwritten digits (0-9) using the MNIST dataset. The model is built using only NumPy, without any deep learning frameworks like TensorFlow or PyTorch.

## Project Structure

- **train_digit.csv**: Training dataset with 42,000 images.
- **test_digit.csv**: Test dataset with 28,000 images.
- **nn_digit.py**: Main script for training the neural network.
- **test.py**: Script for visualizing test images.
- **README.md**: Project documentation.

## Features

- Neural network with one hidden layer.
- ReLU activation for hidden layers and softmax for output.
- Backpropagation implemented for weight and bias updates.
- Model trained with gradient descent.
- Prediction functionality for unseen data.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib

## Installation

1. Clone the repository:
