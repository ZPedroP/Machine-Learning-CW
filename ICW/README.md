# Coronary Heart Disease Classification (2024/25)

This project was developed as part of the MSc Mathematical Trading and Finance programme at Bayes Business School (formerly Cass). It investigates whether a linear or nonlinear decision boundary best classifies Coronary Heart Disease (CHD) in a high-risk male population from the Western Cape, South Africa.

## Overview

Using a dataset of 462 patients with nine clinical, demographic, and lifestyle attributes, this project aims to:

1. Identify the most effective machine learning model for classifying CHD.
2. Investigate whether a linear or nonlinear decision boundary is more appropriate.
3. Explore how techniques like PCA and SMOTE influence model performance.

## Methodology

- **Data Cleaning & Preparation**: The dataset is read from CSV with proper headers, and rows with missing values are dropped.
- **Feature Engineering**:
  - Numerical features are scaled depending on their distribution:
    - Normally distributed → StandardScaler
    - Highly skewed → Log + StandardScaler
    - Bounded (0–1) → MinMaxScaler
  - Categorical variables are one-hot encoded.
- **Optional Dimensionality Reduction**: Principal Component Analysis (PCA) is optionally applied to retain 95% of the variance.
- **Optional Oversampling**: SMOTE is used optionally to address class imbalance.
- **Train-Test Split**: Data is split into 70% training and 30% testing using a fixed random state.
- **Model Training & Tuning**:
  - Models are trained using GridSearchCV to optimise hyperparameters.
  - Accuracy, F1-score, ROC-AUC, and confusion matrices are computed.
  - Final predictions, confusion matrices, and ROC plots are saved.
- **Visualisation**: Correlation heatmaps, distribution plots, and ROC curves are generated and saved for deeper analysis.

## Models Implemented

- Logistic Regression (L2 penalty)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Support Vector Machine (SVM) with RBF and linear kernels
- Decision Tree (tuned using `ccp_alpha`)
- Random Forest (tuned using `max_features`)
- AdaBoost (tuned using `learning_rate`)
- Gradient Boosting (tuned using `learning_rate`)
- k-Nearest Neighbours (tuned using `n_neighbors`)
- Gaussian Naive Bayes

## Results Summary

The models are evaluated on their testing accuracy, F1-score, ROC-AUC, and confusion matrices. Results are automatically saved to the `results/model_performance/` directory.

LDA achieved the highest performance overall, followed closely by Logistic Regression and SVM, confirming that linear decision boundaries are well-suited to this dataset. ROC curve comparisons further illustrate the relative trade-offs in classifier sensitivity and specificity across four different preprocessing configurations:

1. **Baseline** – Standard preprocessing without PCA or SMOTE.
2. **PCA Only** – PCA applied after feature preprocessing. Slight accuracy changes were observed, with minimal performance loss.
3. **SMOTE Only** – SMOTE applied to balance classes. Tree-based models (e.g., Random Forest, Decision Tree) showed improved recall and F1-scores.
4. **PCA + SMOTE** – Combined setup showed mixed results, with some simpler models suffering from reduced performance, indicating sensitivity to transformations.

### ROC Curve Comparisons

<div style="display: flex; flex-wrap: wrap; justify-content: center; margin-top: 10px;">

  <div style="flex-basis: 100%; max-width: 500px; margin: 10px; text-align: center;">
    <img src="https://github.com/ZPedroP/Machine-Learning-CW/blob/main/ICW/results/model_performance/combined_roc_curves_2025-03-21_00-27-10.png" style="width: 100%; height: auto;" />
    <p><em>Figure 1: ROC Curves (Baseline)</em></p>
  </div>

  <div style="flex-basis: 100%; max-width: 500px; margin: 10px; text-align: center;">
    <img src="https://github.com/ZPedroP/Machine-Learning-CW/blob/main/ICW/results/model_performance/smote_pca/combined_roc_curves_2025-03-21_00-52-52_smote_pca.png" style="width: 100%; height: auto;" />
    <p><em>Figure 2: ROC Curves with SMOTE and PCA</em></p>
  </div>

  <div style="flex-basis: 100%; max-width: 500px; margin: 10px; text-align: center;">
    <img src="https://github.com/ZPedroP/Machine-Learning-CW/blob/main/ICW/results/model_performance/pca/combined_roc_curves_2025-03-21_00-51-01_pca.png" style="width: 100%; height: auto;" />
    <p><em>Figure 3: ROC Curves with PCA</em></p>
  </div>

  <div style="flex-basis: 100%; max-width: 500px; margin: 10px; text-align: center;">
    <img src="https://github.com/ZPedroP/Machine-Learning-CW/blob/main/ICW/results/model_performance/smote/combined_roc_curves_2025-03-21_00-46-31_smote.png" style="width: 100%; height: auto;" />
    <p><em>Figure 4: ROC Curves with SMOTE</em></p>
  </div>

</div>

## Repository Structure

```
ICW/
├── datasets/                 # Dataset and data source files
├── images/                   # Raw images and supplementary materials
├── results/                  # Model evaluation results, plots, and metrics
├── README.md                 # Project overview and instructions
├── Report.pdf                # Final report submitted for assessment
├── Task.pdf                  # Coursework brief and task description
├── heart_disease.py          # Main analysis and model training script
└── requirements.txt          # Minimal project dependencies
```

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ZPedroP/Machine-Learning-CW.git
   cd Machine-Learning-CW/ICW
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the main script**:
   ```bash
   python heart_disease.py
   ```

4. **Optional flags**:
   You can enable PCA or SMOTE by modifying these lines in `heart_disease.py`:
   ```python
   smote = 1  # enables class balancing
   pca = 1    # enables PCA dimensionality reduction
   ```

## Requirements

To run this project, the following Python packages are required:

- pandas==2.2.2
- numpy==1.26.4
- matplotlib==3.9.2
- matplotlib-inline==0.1.7
- seaborn==0.13.2
- scikit-learn==1.5.2
- imbalanced-learn==0.13.0
- ISLP==0.4.0

## Author

José Pedro Santos

# Coursework Brief
 
![ICW Cousework for Machine Learning for Quantitative Professionals - Page 1](https://github.com/ZPedroP/Machine-Learning-CW/blob/main/ICW/images/ICW_2025_Page_1.jpg)

![ICW Cousework for Machine Learning for Quantitative Professionals - Page 2](https://github.com/ZPedroP/Machine-Learning-CW/blob/main/ICW/images/ICW_2025_Page_2.jpg)
