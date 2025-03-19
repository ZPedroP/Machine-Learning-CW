import os
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from ISLP import confusion_table
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# --------------------------------------------------
# Data processor to load and preprocess the dataset
# --------------------------------------------------
class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.class_names = None

    def load_data(self):
        # Read CSV using semicolon delimiter and proper header assignment
        column_names = pd.read_csv(self.file_path, nrows=0, delimiter=',').columns.tolist()
        self.df = pd.read_csv(self.file_path, skiprows=1, delimiter=',', names=column_names)
        self.df.dropna(inplace=True) # drop rows with NAs, and replace the original dataframe

        # Features: all but the last 3 columns; Targets: last 3 columns (grades)
        feature_columns = column_names[:-1]
        target_columns = column_names[-1]

        # Split the DataFrame into features (X) and target (y)
        self.X = self.df[feature_columns]
        self.y = self.df[target_columns]

        # TODO: Make sure to address the class imbalance
        print(self.X.dtypes)
        print(self.X.head())
        print(self.y.value_counts())

        return self.X, self.y

    def preprocess(self, X):
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

        # Detect non-normally distributed numerical features (high skewness)
        skewed_features = [col for col in numerical_cols if abs(X[col].skew()) > 1]
        normal_features = [col for col in numerical_cols if col not in skewed_features]

        # StandardScaler for normally distributed features
        normal_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # PowerTransformer (log transform) for skewed numerical features
        skewed_transformer = Pipeline(steps=[
            ('log', FunctionTransformer(np.log1p, validate=True, feature_names_out="one-to-one")),  # log(x+1) to handle zeros
            ('scaler', StandardScaler())
        ])

        # MinMaxScaler for bounded numerical features (e.g., percentages)
        bounded_features = [col for col in numerical_cols if X[col].min() >= 0 and X[col].max() <= 1]
        bounded_transformer = Pipeline(steps=[
            ('scaler', MinMaxScaler())
        ])

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine all transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('normal', normal_transformer, normal_features),
                ('skewed', skewed_transformer, skewed_features),
                ('bounded', bounded_transformer, bounded_features),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

        # Fit the preprocessing pipeline on the data
        preprocessor.fit(X)

        # Transform the data
        X_preprocessed = preprocessor.transform(X)

        # Get the feature names after one-hot encoding
        feature_names = preprocessor.get_feature_names_out()

        print(feature_names)
        # Convert the transformed data back to DataFrames
        X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names, index=X.index)

        return X_preprocessed_df


# -------------------
# Decision Tree Model
# -------------------
class DecisionTreeModel:
    def __init__(self, X_train, y_train, X_test, y_test, random_state=67):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.best_estimator_ = None
        self.model_description = "Decision trees start with the entire dataset and recursively choose the best feature and corresponding threshold that minimises impurity, measured by the Gini Index. Each split divides the data into two subregions, making the best local decision at each step. This results in a tree structure with internal nodes where decisions are made, branches representing pathways from one decision to the next, and terminal nodes which provide the final predictions."

    def fine_tune(self, param_grid={"ccp_alpha": [1, 0.1, 0.01, 0.001, 0.0001]}, cv=20):
        grid = GridSearchCV(DecisionTreeClassifier(random_state=self.random_state),
                            param_grid, scoring='accuracy', cv=cv)

        grid.fit(self.X_train, self.y_train)

        self.best_estimator_ = grid.best_estimator_
        print("Decision Tree Best Params:", grid.best_params_)

        for mean, std, params in zip(grid.cv_results_["mean_test_score"],
                                     grid.cv_results_["std_test_score"],
                                     grid.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))

        return self.best_estimator_

    def make_prediction(self):
        if self.best_estimator_ is None:
            raise ValueError("Model not tuned. Call fine_tune() first.")

        # Predict on the training set
        y_train_pred = self.best_estimator_.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print("Decision Tree Training Accuracy:", train_accuracy)

        # Predict on the test set
        y_test_pred = self.best_estimator_.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print("Decision Tree Testing Accuracy:", test_accuracy)

        fig, ax = plt.subplots()
        plot_tree(self.best_estimator_, feature_names=self.X_train.columns, class_names=['0', '1'], filled=True, ax=ax)
        ax.set_title(f"Decision Tree")

        confusion_table(y_test_pred, self.y_test)

        return train_accuracy, test_accuracy


# -------------------
# Random Forest Model
# -------------------
class RandomForestModel:
    def __init__(self, X_train, y_train, X_test, y_test, n_estimators=500, random_state=0):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.best_estimator_ = None
        self.model_description = "Random forests sample multiple trees (i.e. a forest) using bootstrapped training data (i.e. random). Each tree produces an independent prediction, and the final output is determined by majority voting. To reduce correlation among trees, random forests introduce additional randomness: when splitting a node, each tree considers only a random subset of features (often m=sqrt(p)), where p is the number of features in the dataset (James et al. 2013)."

    def fine_tune(self, param_grid={"max_features": [5, 10, 20, 30, 40, 50, "sqrt"]}, cv=10):
        grid = GridSearchCV(RandomForestClassifier(n_estimators=self.n_estimators, bootstrap=True,
                                                     oob_score=True, random_state=self.random_state),
                            param_grid, scoring='accuracy', cv=cv)

        grid.fit(self.X_train, self.y_train)

        self.best_estimator_ = grid.best_estimator_
        print("Random Forest Best Params:", grid.best_params_)

        for mean, std, params in zip(grid.cv_results_["mean_test_score"],
                                     grid.cv_results_["std_test_score"],
                                     grid.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))

        return self.best_estimator_

    def make_prediction(self):
        if self.best_estimator_ is None:
            raise ValueError("Model not tuned. Call tune() first.")

        # Predict on the training set
        y_train_pred = self.best_estimator_.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print("Random Forest Training Accuracy:", train_accuracy)

        # predict test set labels
        y_test_pred = self.best_estimator_.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print("Random Forest Testing Accuracy:", test_accuracy)

        confusion_table(y_test_pred, self.y_test)

        return train_accuracy, test_accuracy

    def plot_feature_importance(self):
        importances = self.best_estimator_.feature_importances_

        # plot the most important features
        indices = np.argsort(importances)
        top_importances = pd.Series(importances[indices[-10:]], index=self.best_estimator_.feature_names_in_[indices[-10:]])

        fig, ax = subplots()

        top_importances.plot.bar(ax=ax)
        ax.set_ylabel("Mean decrease in impurity")

        plt.tight_layout()
        plt.show()


# --------------
# AdaBoost Model
# --------------
class AdaBoostModel:
    def __init__(self, X_train, y_train, X_test, y_test, n_estimators=100, random_state=0):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.best_estimator_ = None
        self.model_description = "AdaBoost constructs a strong learner by combining multiple weak learners. It adjusts the sample weights based on each weak learner’s performance, giving more focus to misclassified samples in subsequent iterations. Additionally, it assigns weights to weak learners according to their performance and ultimately combines them into a more powerful classifier."

    def fine_tune(self, param_grid={"learning_rate": [0.001, 0.01, 0.1, 1]}, cv=10):
        base_estimator = DecisionTreeClassifier(max_depth=3)
        ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=self.n_estimators, random_state=self.random_state)
        grid = GridSearchCV(ada, param_grid, scoring='accuracy', cv=cv)

        grid.fit(self.X_train, self.y_train)

        self.best_estimator_ = grid.best_estimator_
        print("AdaBoost Best Params:", grid.best_params_)
        for mean, std, params in zip(grid.cv_results_["mean_test_score"],
                                     grid.cv_results_["std_test_score"],
                                     grid.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))

        return self.best_estimator_

    def make_prediction(self):
        if self.best_estimator_ is None:
            raise ValueError("Model not tuned. Call tune() first.")

        # Predict on the training set
        y_train_pred = self.best_estimator_.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print("AdaBoost Training Accuracy:", train_accuracy)

        y_test_pred = self.best_estimator_.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print("AdaBoost Testing Accuracy:", test_accuracy)

        confusion_table(y_test_pred, self.y_test)

        return train_accuracy, test_accuracy
    
    def plot_feature_importance(self):
        importances = self.best_estimator_.feature_importances_

        # plot the most important features
        indices = np.argsort(importances)
        # TODO: Not sure about these indices. Check all the functions that used this implementation
        top_importances = pd.Series(importances[indices[-10:]], index=self.best_estimator_.feature_names_in_[indices[-10:]])

        fig, ax = subplots()

        top_importances.plot.bar(ax=ax)
        ax.set_ylabel("Mean decrease in impurity")

        plt.tight_layout()
        plt.show()


# -----------------------
# Gradient Boosting Model
# -----------------------
class GradientBoostingModel:
    def __init__(self, X_train, y_train, X_test, y_test, n_estimators=100, max_depth=3, random_state=0):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.best_estimator_ = None
        self.model_description = "Gradient Boosting focuses on iteratively correcting the errors of the previous model. It calculates the residuals of the current model, trains a new model to predict these residuals, and then adds the new model’s predictions to the current model, gradually improving overall predictive performance."

    def fine_tune(self, param_grid={"learning_rate": [0.001, 0.01, 0.1, 1]}, cv=10):
        gb = GradientBoostingClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators,
                                        random_state=self.random_state)
        grid = GridSearchCV(gb, param_grid, scoring='accuracy', cv=cv)

        grid.fit(self.X_train, self.y_train)

        self.best_estimator_ = grid.best_estimator_
        print("Gradient Boosting Best Params:", grid.best_params_)
        for mean, std, params in zip(grid.cv_results_["mean_test_score"],
                                     grid.cv_results_["std_test_score"],
                                     grid.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))

        return self.best_estimator_

    def make_prediction(self):
        if self.best_estimator_ is None:
            raise ValueError("Model not tuned. Call tune() first.")

        # Predict on the training set
        y_train_pred = self.best_estimator_.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print("Gradient Boosting Training Accuracy:", train_accuracy)

        # Predict on the test set
        y_test_pred = self.best_estimator_.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print("Gradient Boosting Testing Accuracy:", test_accuracy)

        confusion_table(y_test_pred, self.y_test)

        return train_accuracy, test_accuracy

    def plot_feature_importance(self):
        importances = self.best_estimator_.feature_importances_

        # plot the most important features
        indices = np.argsort(importances)
        top_importances = pd.Series(importances[indices[-10:]], index=self.best_estimator_.feature_names_in_[indices[-10:]])

        fig, ax = subplots()

        top_importances.plot.bar(ax=ax)
        ax.set_ylabel("Mean decrease in impurity")

        plt.tight_layout()
        plt.show()


# -----------------------
# Logistic Regression Model
# -----------------------
class LogisticRegressionModel:
    def __init__(self, X_train, y_train, X_test, y_test, penalty='l2', C=1.0, max_iter=1000, random_state=0):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.best_estimator_ = None
        self.model_description = ""

    def fine_tune(self, param_grid={"C": np.logspace(-5, 1, 300)}, cv=20):
        lr = LogisticRegression(penalty=self.penalty, max_iter=self.max_iter, random_state=self.random_state)
        grid = GridSearchCV(lr, param_grid, scoring='accuracy', cv=cv)

        grid.fit(self.X_train, self.y_train)

        self.best_estimator_ = grid.best_estimator_
        print("Logistic Regression Best Params:", grid.best_params_)
        for mean, std, params in zip(grid.cv_results_["mean_test_score"],
                                     grid.cv_results_["std_test_score"],
                                     grid.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))

        return self.best_estimator_

    def make_prediction(self):
        if self.best_estimator_ is None:
            raise ValueError("Model not tuned. Call tune() first.")

        # Predict on the training set
        y_train_pred = self.best_estimator_.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print("Logistic Regression Training Accuracy:", train_accuracy)

        # Predict on the test set
        y_test_pred = self.best_estimator_.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print("Logistic Regression Testing Accuracy:", test_accuracy)

        confusion_table(y_test_pred, self.y_test)

        return train_accuracy, test_accuracy


# -----------------------
# kNN Model
# -----------------------
class kNNModel:
    def __init__(self, X_train, y_train, X_test, y_test, n_neighbors=5):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_neighbors = n_neighbors
        self.best_estimator_ = None
        self.model_description = ""

    def fine_tune(self, param_grid={"n_neighbors": np.arange(1, 21)}, cv=20):
        knn = KNeighborsClassifier()
        grid = GridSearchCV(knn, param_grid, scoring='roc_auc', cv=cv)

        grid.fit(self.X_train, self.y_train)

        self.best_estimator_ = grid.best_estimator_
        print("kNN Best Params:", grid.best_params_)
        for mean, std, params in zip(grid.cv_results_["mean_test_score"],
                                     grid.cv_results_["std_test_score"],
                                     grid.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))

        return self.best_estimator_

    def make_prediction(self):
        if self.best_estimator_ is None:
            raise ValueError("Model not tuned. Call tune() first.")

        # Predict on the training set
        y_train_pred = self.best_estimator_.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print("kNN Training Accuracy:", train_accuracy)

        # Predict on the test set
        y_test_pred = self.best_estimator_.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print("kNN Testing Accuracy:", test_accuracy)

        confusion_table(y_test_pred, self.y_test)

        return train_accuracy, test_accuracy


# -----------------------
# SVM Model
# -----------------------
class SVMModel:
    def __init__(self, X_train, y_train, X_test, y_test, kernel='rbf', C=1.0, gamma='scale'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.best_estimator_ = None
        self.model_description = ""

    def fine_tune(self, param_grid={"kernel": ["rbf", "linear"], "gamma": [1,1e-1,1e-2,1e-3, 1e-4], "C": [1, 10, 100,1000]}, cv=10):
        svc = SVC()
        grid = GridSearchCV(svc, param_grid, scoring='accuracy', cv=cv)

        grid.fit(self.X_train, self.y_train)

        self.best_estimator_ = grid.best_estimator_
        print("SVM Best Params:", grid.best_params_)
        for mean, std, params in zip(grid.cv_results_["mean_test_score"],
                                     grid.cv_results_["std_test_score"],
                                     grid.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))

        return self.best_estimator_

    def make_prediction(self):
        if self.best_estimator_ is None:
            raise ValueError("Model not tuned. Call tune() first.")

        # Predict on the training set
        y_train_pred = self.best_estimator_.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print("SVM Training Accuracy:", train_accuracy)

        # Predict on the test set
        y_test_pred = self.best_estimator_.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print("SVM Testing Accuracy:", test_accuracy)

        confusion_table(y_test_pred, self.y_test)

        return train_accuracy, test_accuracy


# -----------------------
# Gaussian Naive Bayes Model
# -----------------------
class GaussianNBModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_description = ""
        self.best_estimator_ = None

    def make_prediction(self):
        gnb = GaussianNB()

        gnb.fit(self.X_train, self.y_train)

        # Predict on the training set
        y_train_pred = gnb.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print("GaussianNB Training Accuracy:", train_accuracy)

        # Predict on the test set
        y_test_pred = gnb.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print("GaussianNB Testing Accuracy:", test_accuracy)

        self.best_estimator_ = gnb

        confusion_table(y_test_pred, self.y_test)

        return train_accuracy, test_accuracy


def save_evaluation_metrics(models, X_test, y_test, smote=0):
    """
    Evaluates each model in the given dictionary and saves overall as well as detailed evaluation metrics.
    
    For each model, this function:
      - Calls the model's predict() method to obtain training and test accuracy.
      - Retrieves the tuned (best) estimator's parameters.
      - Computes the F1-score (rounded to three decimals) and saves it to a text file.
      - Computes the confusion matrix and saves it as a CSV file.
      - Computes the ROC curve and AUC, plots the ROC curve, and saves it as a PNG image.
      - Collects overall performance metrics and saves them in a timestamped CSV file.
    
    Parameters:
      models : dict
          Dictionary with model names as keys and model instances as values.
      X_test : array-like
          Test set features.
      y_test : array-like
          Test set labels.
      smote : int, optional (default=0)
          Flag indicating whether SMOTE was applied (affects the output filename).
    """

    # Directory for saving detailed evaluation outputs
    results_dir = "./results/model_performance/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    overall_results = []  # Will hold overall metrics for each model

    # Loop over each model to perform evaluation and save outputs
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")

        # --- Overall Metrics: Training & Testing Accuracy, Best Parameters ---
        try:
            train_acc, test_acc = model.make_prediction()  # Assumes predict() prints info and returns accuracies
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            train_acc, test_acc = "N/A", "N/A"

        if hasattr(model, "best_estimator_") and model.best_estimator_ is not None:
            best_params = model.best_estimator_.get_params()
            y_test_pred = model.best_estimator_.predict(X_test)
        else:
            best_params = "N/A"
            y_test_pred = model.estimator_.predict(X_test)

        # Compute F1-score (binary classification; adjust pos_label or average if needed)
        f1_val = f1_score(y_test, y_test_pred, pos_label=1)

        overall_results.append({
            "Model": model_name,
            "Training Accuracy": round(train_acc, 3) if isinstance(train_acc, float) else train_acc,
            "Testing Accuracy": round(test_acc, 3) if isinstance(test_acc, float) else test_acc,
            "F1-Score": round(f1_val, 3) if isinstance(f1_val, float) else f1_val,
            "Best Parameters": best_params
        })

        # Compute and save the confusion matrix as CSV
        cm = confusion_matrix(y_test, y_test_pred)
        cm_df = pd.DataFrame(cm, index=["Actual Negative", "Actual Positive"],
                             columns=["Predicted Negative", "Predicted Positive"])
        cm_filename = os.path.join(results_dir, f"confusion_table_{model_name}.csv")
        cm_df.to_csv(cm_filename, index=True)
        print(f"Confusion matrix for {model_name} saved to {cm_filename}")

        # Compute ROC curve and AUC.
        # Use predict_proba if available; otherwise, decision_function or fallback to predictions.
        if hasattr(model.best_estimator_, "predict_proba"):
            y_scores = model.best_estimator_.predict_proba(X_test)[:, 1]
        elif hasattr(model.best_estimator_, "decision_function"):
            y_scores = model.best_estimator_.decision_function(X_test)
        else:
            y_scores = model.best_estimator_.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='gray')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc="lower right")
        roc_filename = os.path.join(results_dir, f"roc_curve_{model_name}.png")
        plt.savefig(roc_filename)
        plt.close()
        print(f"ROC curve for {model_name} saved to {roc_filename}")

    # --- Save Overall Results to a Timestamped CSV File ---
    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    overall_filename = f"model_results_{date_time_str}{'_smote' if smote else ''}.csv"
    overall_path = os.path.join(results_dir, overall_filename)
    with open(overall_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Model", "Training Accuracy", "Testing Accuracy", "F1-Score", "Best Parameters"])
        writer.writeheader()
        writer.writerows(overall_results)
    print(f"Overall model results saved to {overall_filename}")


def plot_and_save_feature_distribution(df):
    """
    Plots the distribution of all features in the dataset.
    
    Parameters:
    - df: DataFrame containing the dataset.
    """
    save_dir = "./results/images"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for column in df.columns:
        plt.figure(figsize=(10, 5))
        if df[column].dtype == 'object':  # Categorical features
            df[column].value_counts().plot(kind='bar')
            plt.title(f"Distribution of {column}")
            plt.ylabel("Count")
        else:  # Numerical features
            df[column].hist(bins=30)
            plt.title(f"Distribution of {column}")
            plt.ylabel("Frequency")

        plt.xlabel(column)
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(save_dir, f"{column}_distribution.png")
        plt.savefig(plot_filename)
        plt.close()

    print(f"Feature distribution plots saved in '{save_dir}'")


if __name__ == "__main__":
    # Data loading and preprocessing
    file_path = './datasets/heart-disease.csv'
    smote = 0

    data_processor = DataProcessor(file_path)

    X, y = data_processor.load_data()
    plot_and_save_feature_distribution(X)
    X_preprocessed_df = data_processor.preprocess(X)
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_df, y, test_size=0.3, random_state=42)

    if smote:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

        print(y_train.value_counts())

    # Decision Tree
    dt_model = DecisionTreeModel(X_train, y_train, X_test, y_test)
    dt_model.fine_tune()
    dt_model.make_prediction()

    # Random Forest
    rf_model = RandomForestModel(X_train, y_train, X_test, y_test)
    rf_model.fine_tune()
    rf_model.make_prediction()
    # rf_model.plot_feature_importance()

    # AdaBoost
    ada_model = AdaBoostModel(X_train, y_train, X_test, y_test)
    ada_model.fine_tune()
    ada_model.make_prediction()
    # ada_model.plot_feature_importance()

    # Gradient Boosting
    gb_model = GradientBoostingModel(X_train, y_train, X_test, y_test)
    gb_model.fine_tune()
    gb_model.make_prediction()
    # gb_model.plot_feature_importance()

    # Logistic Regression
    lr_model = LogisticRegressionModel(X_train, y_train, X_test, y_test)
    lr_model.fine_tune()
    lr_model.make_prediction()

    # kNN
    kNN_model = kNNModel(X_train, y_train, X_test, y_test)
    kNN_model.fine_tune()
    kNN_model.make_prediction()

    # SVM
    SVM_model = SVMModel(X_train, y_train, X_test, y_test)
    SVM_model.fine_tune()
    SVM_model.make_prediction()

    # GNB
    GNB_model = GaussianNBModel(X_train, y_train, X_test, y_test)
    GNB_model.make_prediction()

    # Dictionary to store models
    models = {
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "AdaBoost": ada_model,
        "Gradient Boosting": gb_model,
        "Logistic Regression": lr_model,
        "kNN": kNN_model,
        "SVM": SVM_model,
        "GaussianNB": GNB_model
    }

    # Evaluate models and save detailed performance metrics
    save_evaluation_metrics(models, X_test, y_test, smote)
