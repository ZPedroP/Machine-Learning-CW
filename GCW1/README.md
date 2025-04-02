# Decision Trees and Random Forest Project (2024/25)

This project was developed as part of the MSc Mathematical Trading and Finance programme at Bayes Business School (formerly Cass). It received a strong distinction. The work applies classification algorithms, specifically Decision Trees and Random Forests, to predict secondary school student performance using demographic, social, and school-related features. 

The results and methodology are presented through an interactive **Shiny app**, designed to explain model structure and performance to a non-technical audience.

## Overview

The dataset originates from Cortez & Silva (2008), and includes information on Portuguese secondary school students. In accordance with coursework constraints, we restrict our analysis to the Mathematics subset (395 observations), excluding the larger Portuguese-language sample.

Key stages of the project:

1. **Feature Engineering** – Categorical features were one-hot encoded. G1 and G2 (intermediate grade assessments) were excluded due to their high correlation with final grade (G3).
2. **Binary Classification** – Final grades (G3) were grouped into ‘pass’ (G3 ≥ 10) and ‘fail’ (G3 < 10), consistent with the Portuguese grading system.
3. **Model Building** – Decision Tree and Random Forest models were constructed and optimised via cross-validation and grid search.
4. **Shiny App Deployment** – Users can interactively adjust hyperparameters and visualise model structure, training/test performance, and individual decision paths.

## Repository Structure

```
Decision-Tree-and-Random-Forest-Project/
├── app/                             # Shiny app implementation
├── Decision_Tree_RF_Model.ipynb     # Core model development notebook
├── dataset/                         # Cleaned student performance data
├── Images/                          # Visualisations and performance plots
├── Task.pdf                         # Coursework brief
├── Decision Tree and Random Forest for Student Performance Prediction.pdf  # Final report
└── README.md
```

## Summary of Results

| Model           | Cross-Validation Accuracy | Test Accuracy | Comments                         |
|----------------|---------------------------|---------------|----------------------------------|
| Decision Tree  | 70.1%                     | 72.27%        | Best generalisation performance  |
| Random Forest  | 72.7%                     | 66.39%        | Higher variance; prone to overfit|

> ROC curves and accuracy plots confirm that the Decision Tree model generalises better on unseen data. Shapley values identify absences and family dynamics as key predictors.

## Interactive App

The Shiny app allows users to:

- Toggle between Decision Tree and Random Forest visualisations
- Adjust hyperparameters (e.g., α, Max Depth, Number of Estimators)
- View training vs test accuracy in real time
- Explore the structure of individual decision trees

The app is aimed at non-technical stakeholders interested in understanding how model structure impacts predictive performance.

## How to Run

Clone the repository and run the model or launch the app:

```bash
git clone https://github.com/RemaniSA/Decision-Tree-and-Random-Forest-Project.git
cd Decision-Tree-and-Random-Forest-Project
pip install -r requirements.txt  # or install packages individually
```

To explore the Shiny app (Python):

```bash
cd app/
shiny run --reload app.py
```

## Methods

- **Preprocessing**: One-hot encoding, feature engineering, outlier trimming
- **Model Selection**: GridSearchCV with 10-fold cross-validation
- **Evaluation Metrics**: Accuracy, Weighted F1, ROC curves, Shapley values

## Authors

- Shaan Ali Remani  
- José Pedro Pessoa Dos Santos  
- Chin-Lan Chen  
- Poh Har Yap

# Coursework Brief
 
![GCW3 Cousework for Machine Learning for Quantitative Professionals](https://github.com/ZPedroP/Machine-Learning-CW/blob/main/GCW1/images/Task.jpg)
