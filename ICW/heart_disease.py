import os
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from ISLP import confusion_table
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
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

        # Print the class distribution
        print(self.y.value_counts())

        return self.X, self.y

    def preprocess(self, X, apply_pca=0):
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

        if apply_pca:
            # Apply PCA
            pca = PCA(n_components=0.95)
            X_pca = pca.fit_transform(X_preprocessed)
            print("Explained variance ratio:", pca.explained_variance_ratio_)

            # Convert PCA output to DataFrame with generic names: PC1, PC2, ...
            pc_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
            X_preprocessed_pca_df = pd.DataFrame(X_pca, columns=pc_names, index=X.index)

            return X_preprocessed_pca_df
        else:
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
        # Initialize training and testing data, and set random state for reproducibility
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.best_estimator_ = None

    # Fine-tunes the decision tree using cross-validation to find the best hyperparameters
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

    # Make predictions using the trained decision tree and evaluates its performance
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

        # Plot the decision tree
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

    # Fine-tunes the random forest by selecting the best number of features for splitting
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

    # Make predictions using the trained random forest and evaluates its performance
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

    # Fine-tunes the AdaBoost model by adjusting the learning rate
    def fine_tune(self, param_grid={"learning_rate": [0.001, 0.01, 0.1, 1]}, cv=10):
        base_estimator = DecisionTreeClassifier(max_depth=3)
        ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=self.n_estimators, random_state=self.random_state, algorithm='SAMME')
        grid = GridSearchCV(ada, param_grid, scoring='accuracy', cv=cv)

        grid.fit(self.X_train, self.y_train)

        self.best_estimator_ = grid.best_estimator_
        print("AdaBoost Best Params:", grid.best_params_)
        for mean, std, params in zip(grid.cv_results_["mean_test_score"],
                                     grid.cv_results_["std_test_score"],
                                     grid.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))

        return self.best_estimator_

    # Make predictions using the trained AdaBoost model and evaluates its performance
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

    # Fine-tunes the gradient boosting model by selecting the best learning rate
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

    # Make predictions using the trained gradient boosting model and evaluates its performance
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

    # Fine-tunes logistic regression by selecting the optimal regularization parameter
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

    # Make predictions using the trained logistic regression model and evaluates its performance
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

    # Fine-tunes kNN by selecting the optimal number of neighbours
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

    # Make predictions using the trained kNN model and evaluates its performance
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

    # Fine-tunes the SVM by selecting the best kernel, regularization parameter, and gamma
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

    # Make predictions using the trained SVM model and evaluates its performance
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
        self.best_estimator_ = None

    # Train and make predictions using the Gaussian Naive Bayes model, then evaluate performance
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

        # Save model as a class attribute
        self.best_estimator_ = gnb

        confusion_table(y_test_pred, self.y_test)

        return train_accuracy, test_accuracy


# -----------------------
# LDA Model
# -----------------------
class LDAModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_estimator_ = None

    # Train and make predictions using the LDA model, then evaluate performance
    def make_prediction(self):
        lda = LinearDiscriminantAnalysis()

        lda.fit(self.X_train, self.y_train)

        # Predict on the training set
        y_train_pred = lda.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print("LDA Training Accuracy:", train_accuracy)

        # Predict on the test set
        y_test_pred = lda.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print("LDA Testing Accuracy:", test_accuracy)

        # Save model as a class attribute
        self.best_estimator_ = lda

        confusion_table(y_test_pred, self.y_test)

        return train_accuracy, test_accuracy


# -----------------------
# QDA Model
# -----------------------
class QDAModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_estimator_ = None

    # Train and make predictions using the QDA model, then evaluate performance
    def make_prediction(self):
        qda = QuadraticDiscriminantAnalysis()

        qda.fit(self.X_train, self.y_train)

        # Predict on the training set
        y_train_pred = qda.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print("QDA Training Accuracy:", train_accuracy)

        # Predict on the test set
        y_test_pred = qda.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print("QDA Testing Accuracy:", test_accuracy)

        # Save model as a class attribute
        self.best_estimator_ = qda

        confusion_table(y_test_pred, self.y_test)

        return train_accuracy, test_accuracy


def save_evaluation_metrics(models, X_test, y_test, smote=0, pca=0):
    """
    Evaluates each model in the provided dictionary and saves detailed evaluation metrics.
    
    For each model, this function:
      - Calls the model's make_prediction() method to obtain training and test accuracy.
      - Retrieves the tuned estimator's parameters (if available) or falls back to an alternative estimator.
      - Computes the F1-score (rounded to three decimals) for the positive class (assumed to be labeled as 1).
      - Computes and saves the confusion matrix as a CSV file.
      - Computes the ROC curve and AUC, plots the ROC curve, and saves it as a PNG image.
      - Collects performance metrics and saves them in a timestamped CSV file.
    
    Parameters:
      models : dict
          Dictionary with model names as keys and model instances as values.
      X_test : array-like
          Test set features.
      y_test : array-like
          Test set labels.
      smote : bool, optional (default=False)
          Flag indicating whether SMOTE was applied (affects the output filename).
      pca : bool, optional (default=False)
          Flag indicating whether PCA was applied (affects output filenames).
    
    Returns:
      None
    """

    # Get current timestamp for unique file naming
    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Determine results directory based on preprocessing techniques applied
    if smote and not pca:
        results_dir = "./results/model_performance/smote/"
    elif pca and not smote:
        results_dir = "./results/model_performance/pca/"
    elif smote and pca:
        results_dir = "./results/model_performance/smote_pca/"
    else:
        results_dir = "./results/model_performance/"

    # Create the results directory if it does not exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # List to store performance metrics for each model
    results = []
    # Dictionary to store ROC curve data for combined plot
    all_roc_data = {}

    # Loop over each model to evaluate and save performance metrics
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")

        # Evaluate model performance using its make_prediction method
        try:
            train_acc, test_acc = model.make_prediction()
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            train_acc, test_acc = "N/A", "N/A"

        # Retrieve the best estimator and predict on X_test
        if hasattr(model, "best_estimator_") and model.best_estimator_ is not None:
            best_params = model.best_estimator_.get_params()
            y_test_pred = model.best_estimator_.predict(X_test)
        else:
            best_params = "N/A"
            # Fallback: attempt to use 'estimator_' if available
            y_test_pred = model.estimator_.predict(X_test)

        # Compute F1-score for the positive class (assumed label=1)
        f1_val = f1_score(y_test, y_test_pred, pos_label=1)

        # Append model evaluation results to the list
        results.append({
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
        cm_filename = os.path.join(results_dir, f"confusion_table_{model_name}_{date_time_str}{'_smote' if smote else ''}{'_pca' if pca else ''}.csv")
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

        # Store ROC data for the combined plot
        all_roc_data[model_name] = (fpr, tpr, roc_auc)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='gray')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc="lower right", fontsize=6)
        roc_filename = os.path.join(results_dir, f"roc_curve_{model_name}_{date_time_str}{'_smote' if smote else ''}{'_pca' if pca else ''}.png")
        plt.savefig(roc_filename, dpi=300)
        plt.close()
        print(f"ROC curve for {model_name} saved to {roc_filename}")

    # --- Combined ROC Plot for All Models ---
    plt.figure()
    for model_name, (fpr, tpr, roc_auc) in all_roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='gray')  # Diagonal line for random performance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves for All Models')
    plt.legend(loc="lower right", fontsize=6)
    combined_roc_filename = os.path.join(results_dir, f"combined_roc_curves_{date_time_str}"f"{'_smote' if smote else ''}{'_pca' if pca else ''}.png")
    plt.savefig(combined_roc_filename, dpi=300)
    plt.close()
    print(f"Combined ROC curve for all models saved to {combined_roc_filename}")

    # --- Save results to CSV file ---
    results_filename = f"models_results_{date_time_str}{'_smote' if smote else ''}{'_pca' if pca else ''}.csv"
    results_path = os.path.join(results_dir, results_filename)
    with open(results_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Model", "Training Accuracy", "Testing Accuracy", "F1-Score", "Best Parameters"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Model results saved to {results_filename}")


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
        plt.savefig(plot_filename, dpi=300)
        plt.close()

    print(f"Feature distribution plots saved in '{save_dir}'")


if __name__ == "__main__":
    # Get current timestamp for unique file naming
    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Set file path for the dataset and flags for applying SMOTE and PCA.
    file_path = './datasets/heart-disease.csv'
    # Set to 1 if SMOTE oversampling should be applied to balance the classes.
    smote = 0
    # Set to 1 if PCA dimensionality reduction should be applied to the features.
    pca = 0

    # Instantiate DataProcessor with the provided file path.
    data_processor = DataProcessor(file_path)

    # Load data from CSV file into DataFrame and split into features (X) and target (y).
    X, y = data_processor.load_data()

    # Optionally, plot and save feature distributions for exploratory analysis.
    # plot_and_save_feature_distribution(X)

    # Preprocess the features. If PCA is enabled, the function applies PCA and returns reduced dimensions.
    X_preprocessed_df = data_processor.preprocess(X, pca)

    # If PCA is not applied, compute and visualize the correlation matrix of the preprocessed features.
    if not pca:
        # Determine results directory based on preprocessing techniques applied
        if smote:
            results_dir = "./results/model_performance/smote/"
        else:
            results_dir = "./results/model_performance/"

        # Calculate the correlation matrix and round values for better readability.
        correlation_table = X_preprocessed_df.corr().round(2)

        # Create a heatmap of the correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_table, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Feature Correlations")
        plt.tight_layout()

        # Save the heatmap as a high-resolution PNG image.
        plt.savefig(results_dir + "correlation_table.png", dpi=300)
        plt.close()

        # Save the correlation table as a CSV file for further analysis.
        correlation_table.to_csv(results_dir + "correlation_table.csv")

    # Split the preprocessed data into training and testing sets (70% training, 30% testing) using a fixed random state for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_df, y, test_size=0.3, random_state=42)

    # If SMOTE oversampling is enabled, apply it to the training set to balance class distribution.
    if smote:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        # Print the class distribution after applying SMOTE to verify the balancing effect.
        print(y_train.value_counts())

    # -----------------------
    # Initialize and evaluate various machine learning models
    # -----------------------

    # Decision Tree Model
    dt_model = DecisionTreeModel(X_train, y_train, X_test, y_test)
    dt_model.fine_tune()
    dt_model.make_prediction()

    # Random Forest Model
    rf_model = RandomForestModel(X_train, y_train, X_test, y_test)
    rf_model.fine_tune()
    rf_model.make_prediction()

    # AdaBoost Model
    ada_model = AdaBoostModel(X_train, y_train, X_test, y_test)
    ada_model.fine_tune()
    ada_model.make_prediction()

    # Gradient Boosting Model
    gb_model = GradientBoostingModel(X_train, y_train, X_test, y_test)
    gb_model.fine_tune()
    gb_model.make_prediction()

    # Logistic Regression Model
    lr_model = LogisticRegressionModel(X_train, y_train, X_test, y_test)
    lr_model.fine_tune()
    lr_model.make_prediction()

    # kNN Model
    kNN_model = kNNModel(X_train, y_train, X_test, y_test)
    kNN_model.fine_tune()
    kNN_model.make_prediction()

    # Support Vector Machine (SVM) Model
    SVM_model = SVMModel(X_train, y_train, X_test, y_test)
    SVM_model.fine_tune()
    SVM_model.make_prediction()

    # Gaussian Naive Bayes Model
    GNB_model = GaussianNBModel(X_train, y_train, X_test, y_test)
    GNB_model.make_prediction()

    # Linear Discriminant Analysis (LDA) Model
    LDA_model = LDAModel(X_train, y_train, X_test, y_test)
    LDA_model.make_prediction()

    # Quadratic Discriminant Analysis (QDA) Model
    QDA_model = QDAModel(X_train, y_train, X_test, y_test)
    QDA_model.make_prediction()

    # Create a dictionary to store all the models for evaluation.
    models = {
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "AdaBoost": ada_model,
        "Gradient Boosting": gb_model,
        "Logistic Regression": lr_model,
        "kNN": kNN_model,
        "SVM": SVM_model,
        "GaussianNB": GNB_model,
        "LDA": LDA_model,
        "QDA": QDA_model
    }

    # Evaluate models and save detailed performance metrics
    save_evaluation_metrics(models, X_test, y_test, smote, pca)
