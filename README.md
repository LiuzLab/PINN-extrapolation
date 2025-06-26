# Code for the paper titled "Improving physics-informed neural network extrapolation via transfer learning and adaptive activation functions"

The paper has been accepted to **ICANN 2025**.

## Overview

In this work, we present a novel approach for improving the extrapolation capabilities of Physics-Informed Neural Networks (PINNs), based on a combination of **Adaptive Activation Functions** and **Transfer Learning**.

## Requirements

To run this code, ensure you have the following dependencies as specified in the `requirements.txt` file.

## Installation

To install the required dependencies, clone the repository and install the necessary Python packages:

```bash
git clone https://github.com/LiuzLab/PINN-extrapolation.git
cd PINN-extrapolation
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

