from shiny import ui, render, App
from sklearn.model_selection import train_test_split
from .student-performance import DataProcessor
from .student-performance import DecisionTreeModel
from .student-performance import RandomForestModel
from .student-performance import AdaBoostModel
from .student-performance import GradientBoostingModel

# Data loading and preprocessing
file_path = './datasets/student-mat.csv'
data_processor = DataProcessor(file_path)
X, y = data_processor.load_data()
X_preprocessed_df = data_processor.preprocess(X)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_df, y, test_size=0.3, random_state=42)

print(X_train)