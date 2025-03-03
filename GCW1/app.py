import os
import sys

ROOT_PATH = os.path.dirname(__file__)
print(ROOT_PATH)
#ROOT_PATH = "/".join(ROOT_PATH.split("/")[:-1])
#sys.path.append(ROOT_PATH)

from shiny import ui, render, App
from sklearn.model_selection import train_test_split
from student_performance import DataProcessor
from student_performance import DecisionTreeModel
from student_performance import RandomForestModel
from student_performance import AdaBoostModel
from student_performance import GradientBoostingModel

# Data loading and preprocessing
file_path = ROOT_PATH + '/datasets/student-mat.csv'
data_processor = DataProcessor(file_path)
X, y = data_processor.load_data()
X_preprocessed_df = data_processor.preprocess(X)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_df, y, test_size=0.3, random_state=42)

# Define the user interface (UI)
app_ui = ui.page_fluid(
    ui.h2("Classification App"),

    # Top-level input: switch between Binary or Multiclass
    ui.row(
        ui.column(
            4,
            ui.input_select(
                "classification_type",
                "Classification Type",
                {
                    "Binary": "Binary",
                    "Multiclass": "Multiclass"
                },
                selected="Binary"
            )
        )
    ),

    # Two-column layout for Left and Right "models"
    ui.row(
        # Left Column
        ui.column(
            6,
            ui.h3("Model Description (Left)"),
            ui.output_text("modelDescLeft"),

            ui.input_select(
                "paramSelLeft1",
                "Parameter Selection (Left #1)",
                {"param1": "param1", "param2": "param2", "param3": "param3"}
            ),
            ui.input_select(
                "treeSelLeft",
                "Tree Selection (Left)",
                {"Tree A": "Tree A", "Tree B": "Tree B"}
            ),
            ui.input_select(
                "paramSelLeft2",
                "Parameter Selection (Left #2)",
                {"paramX": "paramX", "paramY": "paramY"}
            ),

            ui.output_plot("treePlotLeft"),

            ui.h4("Results (Left)"),
            ui.output_text("trainScoreLeft"),
            ui.output_text("testScoreLeft")
        ),

        # Right Column
        ui.column(
            6,
            ui.h3("Model Description (Right)"),
            ui.output_text("modelDescRight"),

            ui.input_select(
                "paramSelRight1",
                "Parameter Selection (Right #1)",
                {"param1": "param1", "param2": "param2", "param3": "param3"}
            ),
            ui.input_select(
                "treeSelRight",
                "Tree Selection (Right)",
                {"Tree A": "Tree A", "Tree B": "Tree B"}
            ),
            ui.input_select(
                "paramSelRight2",
                "Parameter Selection (Right #2)",
                {"paramX": "paramX", "paramY": "paramY"}
            ),

            ui.output_plot("treePlotRight"),

            ui.h4("Results (Right)"),
            ui.output_text("trainScoreRight"),
            ui.output_text("testScoreRight")
        )
    )
)

# Define the server logic
def server(input, output, session):
    @output
    @render.text
    def modelDescLeft():
        return (
            f"Selected classification type: {input.classification_type()}\n"
            f"Parameters chosen: {input.paramSelLeft1()} and {input.paramSelLeft2()}\n"
            f"Tree type: {input.treeSelLeft()}"
        )

    @output
    @render.text
    def modelDescRight():
        return (
            f"Selected classification type: {input.classification_type()}\n"
            f"Parameters chosen: {input.paramSelRight1()} and {input.paramSelRight2()}\n"
            f"Tree type: {input.treeSelRight()}"
        )

    # Placeholder for a "tree" plot on the left
    @output
    @render.plot
    def treePlotLeft():
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        ax.plot(np.random.rand(10))
        ax.set_title("Left Model Tree")
        return fig

    # Placeholder for a "tree" plot on the right
    @output
    @render.plot
    def treePlotRight():
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        ax.plot(np.random.rand(10))
        ax.set_title("Right Model Tree")
        return fig

    @output
    @render.text
    def trainScoreLeft():
        return "Train Score (Left): 0.95"

    @output
    @render.text
    def testScoreLeft():
        return "Test Score (Left): 0.90"

    @output
    @render.text
    def trainScoreRight():
        return "Train Score (Right): 0.92"

    @output
    @render.text
    def testScoreRight():
        return "Test Score (Right): 0.88"

# Create the Shiny app object
app = App(app_ui, server)