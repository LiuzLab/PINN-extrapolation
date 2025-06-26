# Code for the paper "Improving physics-informed neural network extrapolation via transfer learning and adaptive activation functions"
Our work has just been accepted to ICANN 2025!

## Overview

Physics-Informed Neural Networks (PINNs) are deep learning models that incorporate the governing physical laws of a system into the learning process, making them well-suited for solving complex scientific and engineering problems. Recently, PINNs have gained widespread attention as a powerful framework for combining physical principles with data-driven modeling to improve prediction accuracy. Despite their successes, however, PINNs often exhibit poor extrapolation performance outside the training domain and are highly sensitive to the choice of activation functions (AFs). In this paper, we introduce a transfer learning (TL) method to improve the extrapolation capability of PINNs. Our approach applies transfer learning (TL) within an extended training domain, using only a small number of carefully selected collocation points. Additionally, we propose an adaptive AF that takes the form of a linear combination of standard AFs, which improves both the robustness and accuracy of the model. Through a series of experiments, we demonstrate that our method achieves an average of 40% reduction in relative $L_2$ error and an average of 50% reduction in mean absolute error in the extrapolation domain, all without a significant increase in computational cost.

## Requirements

To run this code, ensure you have the following dependencies as specified in the `requirements.txt` file.

## Installation

To install the required dependencies, clone the repository and install the necessary Python packages:

```bash
# Create a new conda environment
conda create -n pinn-extrap python=3.11.3

# Activate the conda environment
conda activate pinn-extrap

# Clone the repository
git clone https://github.com/LiuzLab/PINN-extrapolation.git
cd PINN-extrapolation

# Install the required dependencies
pip install -r requirements.txt
```

## Directory Structure

The code is organized into three main directories to accommodate the three different equations from the paper using two configurations for each:

- **AC**: Contains the exact solution data and code for the Allen-Cahn equation.
    - **tanh**: Contains the code using the traditional tanh activation function.
    - **lctanh**: Contains the code that uses the best linear combination of tanh, as described in the paper.
  
- **KdV**: Contains the exact solution data and code for the Korteweg de Vries equation.
    - **tanh**: Contains the code using the traditional tanh activation function.
    - **lcxsinx**: Contains the code that uses the best linear combination of $$x + \sin^2(x)$$ , as described in the paper.
  
- **Burgers**: Contains the exact solution data and code for the Burgers' equation.
    - **tanh**: Contains the code using the traditional tanh activation function.
    - **lctanh**: Contains the code that uses the best linear combination of tanh, as described in the paper.
 
All codes include both the initial training and transfer learning phases.

