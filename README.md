# Linear Regression in Rust: High-Performance Implementation

This repository provides a high-performance implementation of Linear Regression written in Rust. It showcases Rust's speed and efficiency for computationally intensive machine learning tasks, significantly outperforming traditional Python implementations in execution time. For example, this implementation processes 15,000 iterations in approximately 5 seconds on an i7 CPU.

## 🚀 Key Features

*   **Dataset Handling:**
    *   Reads CSV files and parses numeric features and labels.
    *   Provides seamless data extraction for training.

*   **Data Normalization:**
    *   Implements Min-Max Scaling to normalize feature values between 0 and 1.
    *   Improves gradient descent convergence and training speed.

*   **Train-Test Split:**
    *   Splits the dataset into training and testing sets based on a configurable ratio.
    *   Enables accurate model evaluation and generalization analysis.

*   **Gradient Descent Optimization:**
    *   Uses batch gradient descent to iteratively update model coefficients.
    *   Efficiently minimizes the loss function, leveraging Rust's speed and memory safety.

*   **Model Evaluation:**
    *   Calculates predictions on the test dataset.
    *   Computes Mean Squared Error (MSE) to assess model performance.

*   **Performance Visualization (Optional):**
    *   Option to generate a loss plot using the `plotters` crate.
    *   Visualizes the optimization progress over time.

## ⚙️ Technical Implementation

### Dataset Reading

The `csv` crate is used to handle CSV files, enabling seamless extraction of features and labels. The dataset is assumed to have numeric features and labels, which are parsed into floating-point numbers for compatibility with the gradient descent algorithm.

### Data Normalization

Normalization is performed using Min-Max Scaling, which scales feature values to a range of 0 to 1. This step ensures better gradient descent behavior and faster convergence during training.

### Gradient Descent

The implementation uses batch gradient descent to iteratively update coefficients. The algorithm minimizes the loss function efficiently, leveraging Rust's computational speed and memory safety to achieve low execution times.

### Model Evaluation

Model evaluation is performed using Mean Squared Error (MSE) as the primary metric. Predictions are calculated for the test dataset, and the MSE quantifies the error between predicted and actual labels.

### Performance Benchmark

This implementation highlights Rust's superior execution speed. For example, processing 15,000 iterations on an i7 CPU takes approximately 5 seconds, compared to significantly longer runtimes in Python. This advantage demonstrates Rust's efficiency in handling machine learning workloads.

## 🛠️ Requirements

To run the project, ensure you have the following installed:

*   **Rust:** (latest stable version)
    *   [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

*   **Required Crates:**
    *   `csv` crate for dataset handling
        ```toml
        csv = "1.2"
        ```
    *   `rand` crate for shuffling data
        ```toml
        rand = "0.8"
        ```
    *    (Optional) `plotters` crate for generating loss plots
        ```toml
        plotters = "0.3"
        ```
   
