
# MLOps Assignment 1: House Price Prediction

[![Model Training CI](https://github.com/utkarsh-anant/ML_Ops_Assmt_1/actions/workflows/ci.yml/badge.svg?branch=kernelridge)](https://github.com/utkarsh-anant/ML_Ops_Assmt_1/actions/workflows/ci.yml)

An end-to-end Machine Learning project demonstrating core MLOps principles: environment management, code modularity, version control with branching strategies, and CI/CD automation.

## Project Overview

This project implements and compares two different regression models (**Decision Tree** and **Kernel Ridge**) on the Boston Housing dataset. The core focus is on establishing a reproducible and automated pipeline, rather than just achieving the highest accuracy.

**Key Highlights:**
- **Modular Design:** Reusable functions for data handling and training.
- **Git Branching Strategy:** Isolated development for different models.
- **CI/CD Pipeline:** Automated testing and reporting on every push.
- **Performance Comparison:** Automated report to determine the best model.

## Project Structure

# Project Structure

```
ML_Ops_Assmt_1/
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD workflow
│
├── misc.py                        # Core utilities (load, preprocess, train, evaluate)
├── train.py                       # Decision Tree Regressor implementation  
├── train2.py                      # Kernel Ridge Regressor implementation
├── compare_models.py              # Script to generate a model performance report
├── requirements.txt               # Project dependencies
└── README.md
```

## Getting Started

### Prerequisites
- Git
- Conda (Miniconda or Anaconda)

### Installation & Execution

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/utkarsh-anant/ML_Ops_Assmt_1.git
    cd ML_Ops_Assmt_1
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create -n mlops-assignment-1 python=3.9 -y
    conda activate mlops-assignment-1
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the models individually:**
    ```bash
    # Run Decision Tree
    python train.py

    # Run Kernel Ridge
    python train2.py
    ```

5.  **Generate the performance comparison report:**
    ```bash
    python compare_models.py
    ```

## CI/CD Pipeline

This project uses GitHub Actions. The workflow (`.github/workflows/ci.yml`) is triggered on every push or pull request to the `kernelridge` branch. It automatically:

- Sets up a Python 3.9 environment.
- Installs all dependencies from `requirements.txt`.
- Runs `train.py` (Decision Tree).
- Runs `train2.py` (Kernel Ridge).
- Executes `compare_models.py` to generate the final performance report.

You can view the workflow runs and their logs in the [**Actions**](https://github.com/utkarsh-anant/ML_Ops_Assmt_1/actions) tab of this repository.

## Branching Strategy

- `main`: The stable branch containing the final, working code and documentation.
- `dtree`: Contains the implementation of the Decision Tree Regressor.
- `kernelridge`: Contains the implementation of the Kernel Ridge Regressor and the integrated CI/CD pipeline.
