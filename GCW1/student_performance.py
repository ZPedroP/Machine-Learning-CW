import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from ISLP import confusion_table
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# Data processor to load and preprocess the dataset
# --------------------------------------------------

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None

    def load_data(self):
        # Read CSV using semicolon delimiter and proper header assignment
        column_names = pd.read_csv(self.file_path, nrows=0, delimiter=';').columns.tolist()
        self.df = pd.read_csv(self.file_path, skiprows=1, delimiter=';', names=column_names)

        # Features: all but the last 3 columns; Targets: last 3 columns (grades)
        feature_columns = column_names[:-3]
        target_columns = column_names[-3:]

        # Split the DataFrame into features (X) and target (y)
        self.X = self.df[feature_columns]
        y_df = self.df[target_columns]

        # Use only the final grade: drop the first two grade columns and rename the remaining one
        y_df = y_df.drop(y_df.columns[:2], axis=1).rename(columns={y_df.columns[0]: "Final Grade"})
        y_array = np.reshape(y_df, (-1,))

        # Binarize: grade < 10 -> 0 (fail), grade >= 10 -> 1 (pass)
        self.y = np.where(y_array < 10, 0, 1)
        return self.X, self.y

    def preprocess(self, X):
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[('cat', categorical_transformer, categorical_cols)],
            remainder='passthrough'
        )

        # Fit the preprocessing pipeline on the data
        preprocessor.fit(X)

        # Transform the data
        X_preprocessed = preprocessor.transform(X)

        # Get the feature names after one-hot encoding
        feature_names = preprocessor.get_feature_names_out()

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

    def predict(self):
        if self.best_estimator_ is None:
            raise ValueError("Model not tuned. Call fine_tune() first.")

        y_pred = self.best_estimator_.predict(self.X_test)
        print("Decision Tree Accuracy:", accuracy_score(self.y_test, y_pred))

        confusion_table(y_pred, self.y_test)

        return y_pred

    def plot_tree(self, ccp_alpha=0.01, max_depth=None, return_results = 0):
        # Train a decision tree with the chosen complexity parameter and visualize it
        dt = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=self.random_state, max_depth=max_depth)

        dt.fit(self.X_train, self.y_train)

        # Predict on the training set
        y_train_pred = dt.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print("Decision Tree Training Accuracy:", train_accuracy)

        # Predict on the test set
        y_test_pred = dt.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print("Decision Tree Test Accuracy:", test_accuracy)

        fig, ax = plt.subplots()
        plot_tree(dt, feature_names=self.X_train.columns, class_names=['0', '1'], filled=True, ax=ax)
        ax.set_title(f"Decision Tree (learning rate={ccp_alpha})")

        if return_results:
            return train_accuracy, test_accuracy
        else:
            return fig


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

    def predict(self):
        if self.best_estimator_ is None:
            raise ValueError("Model not tuned. Call tune() first.")

        # predict test set labels
        y_pred = self.best_estimator_.predict(self.X_test)
        print("Random Forest Accuracy:", accuracy_score(self.y_test, y_pred))

        confusion_table(y_pred, self.y_test)

        return y_pred

    def plot_feature_importance(self, max_features=10):
        # Train a Random Forest with fixed max_features to extract and plot feature importances
        rf = RandomForestClassifier(n_estimators=self.n_estimators, max_features=max_features,
                                    bootstrap=True, oob_score=True, random_state=self.random_state)

        rf.fit(self.X_train, self.y_train)

        importances = rf.feature_importances_

        # plot the most important features
        indices = np.argsort(importances)
        top_importances = pd.Series(importances[indices[-10:]], index=rf.feature_names_in_[indices[-10:]])

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
        self.model_description = "AdaBoost Model placeholder description"

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

    def predict(self):
        if self.best_estimator_ is None:
            raise ValueError("Model not tuned. Call tune() first.")
        
        y_pred = self.best_estimator_.predict(self.X_test)
        print("AdaBoost Accuracy:", accuracy_score(self.y_test, y_pred))

        confusion_table(y_pred, self.y_test)

        return y_pred
    
    def plot_feature_importance(self):
        # Train an AdaBoost model for feature importance visualization
        base_estimator = DecisionTreeClassifier(max_depth=3)
        ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=self.n_estimators, learning_rate=0.01 , random_state=self.random_state)

        ada.fit(self.X_train, self.y_train)

        importances = ada.feature_importances_

        # plot the most important features
        indices = np.argsort(importances)
        top_importances = pd.Series(importances[indices[-10:]], index=ada.feature_names_in_[indices[-10:]])

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
        self.model_description = "Gradient Boosting Model placeholder description"

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

    def predict(self):
        if self.best_estimator_ is None:
            raise ValueError("Model not tuned. Call tune() first.")

        y_pred = self.best_estimator_.predict(self.X_test)
        print("Gradient Boosting Accuracy:", accuracy_score(self.y_test, y_pred))

        confusion_table(y_pred, self.y_test)

        return y_pred

    def plot_feature_importance(self):
        # Train a Gradient Boosting model with a preset learning_rate (e.g., 0.01) for feature importance visualization
        gb = GradientBoostingClassifier(learning_rate=0.01, max_depth=self.max_depth,
                                        n_estimators=self.n_estimators, random_state=self.random_state)

        gb.fit(self.X_train, self.y_train)

        importances = gb.feature_importances_

        # plot the most important features
        indices = np.argsort(importances)
        top_importances = pd.Series(importances[indices[-10:]], index=gb.feature_names_in_[indices[-10:]])

        fig, ax = subplots()

        top_importances.plot.bar(ax=ax)
        ax.set_ylabel("Mean decrease in impurity")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Data loading and preprocessing
    file_path = './datasets/student-mat.csv'
    data_processor = DataProcessor(file_path)
    X, y = data_processor.load_data()
    X_preprocessed_df = data_processor.preprocess(X)
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_df, y, test_size=0.3, random_state=42)

    # Decision Tree
    dt_model = DecisionTreeModel(X_train, y_train, X_test, y_test)
    dt_model.fine_tune()
    dt_model.predict()
    dt_model.plot_tree()

    # Random Forest
    rf_model = RandomForestModel(X_train, y_train, X_test, y_test)
    rf_model.fine_tune()
    rf_model.predict()
    rf_model.plot_feature_importance()

    # AdaBoost
    ada_model = AdaBoostModel(X_train, y_train, X_test, y_test)
    ada_model.fine_tune()
    ada_model.predict()

    # Gradient Boosting
    gb_model = GradientBoostingModel(X_train, y_train, X_test, y_test)
    gb_model.fine_tune()
    gb_model.predict()
    gb_model.plot_feature_importance()
