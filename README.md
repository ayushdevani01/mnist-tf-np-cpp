# mnist-tf-np-cpp

At first i built neural network in TensorFlow it gave faster training time and great accuracy but i wanted to know what happens when you do model.fit() so ibused forward and backward propagation and made it using scratch using python and at last i made it using c++ using Eigen library for matrix manipulations and logic was almost same as python.I compiled the C++ code with -O3 optimization flag which enables aggressive compiler optimizations.

## The Implementations
I kept the logic identical across all versions:
- **Input**: 784 pixels (28x28 images flattened).
- **Architecture**: 1 hidden layer (10 neurons) + output layer (10 neurons).
- **Activation**: ReLU (hidden) + Softmax (output).
- **Loss**: Cross-entropy.
- **Optimizer**: Stochastic Gradient Descent (SGD).


| Tool          | Total Time (Seconds) | Time/Iteration (Seconds) | Test Accuracy | Iterations |
|---------------|----------------------|--------------------------|---------------|------------|
| **TensorFlow** | 36.12                | 2.41                     | 92.10%        | 15         |
| **Python (NumPy)** | 262.05           | 0.52                     | 81.45%        | 500        |
| **C++ (Eigen)** | 97                   | 0.194                    | 83.94%        | 500        |


