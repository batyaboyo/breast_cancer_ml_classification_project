# Breast Cancer Classification Project

## Overview
This project focuses on using machine learning to classify breast cancer tumors as benign or malignant based on the Breast Cancer Wisconsin dataset. The implementation uses a K-Nearest Neighbors (KNN) classifier in a Jupyter Notebook environment.

## Dataset
The Breast Cancer Wisconsin dataset contains 569 instances with 30 features computed from digitized images of fine needle aspirate (FNA) of breast masses. Features include measurements like radius, texture, perimeter, area, smoothness, and more.

Dataset source: [Kaggle - Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)

## Requirements
- Anaconda
- Jupyter Notebook
- Python libraries:
  - scikit-learn
  - numpy
  - pandas
  - matplotlib
  - seaborn

## Project Tasks
1. **Load the Dataset**: Using scikit-learn's built-in dataset loader
2. **Split the Dataset**: Creating training (80%) and testing (20%) sets
3. **Train a KNN Classifier**: Implementing and training a K-Nearest Neighbors model
4. **Evaluate the Model**: Calculating and interpreting:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
5. **Explain Evaluation Metrics**: Providing mathematical formulas for each metric
6. **Mathematical Verification**: Demonstrating how the test metrics are calculated

## Implementation Steps
1. Import necessary libraries
2. Load the Breast Cancer Wisconsin dataset
3. Explore and preprocess the data if needed
4. Split data into training and testing sets
5. Train the KNN classifier
6. Make predictions on the test set
7. Calculate and visualize evaluation metrics
8. Explain metrics with formulas and calculate them manually

## How to Run
1. Ensure Anaconda is installed
2. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```
3. Open the project notebook
4. Run all cells in sequence

## Project Structure
- `breast_cancer_classification.ipynb`: Main Jupyter Notebook with all code and explanations
- `README.md`: Project documentation

## Evaluation Metrics
The project includes detailed explanations of:
- **Accuracy**: Proportion of correct predictions
- **Precision**: Proportion of positive identifications that were actually correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1-score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Table showing true positives, false positives, true negatives, and false negatives
