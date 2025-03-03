import os
import sys

ROOT_PATH = os.path.dirname(__file__)

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

decision_tree_model = DecisionTreeModel(X_train, y_train, X_test, y_test)
random_forest_model = RandomForestModel(X_train, y_train, X_test, y_test)
ada_boost_model = AdaBoostModel(X_train, y_train, X_test, y_test)
gradient_boosting_model = GradientBoostingModel(X_train, y_train, X_test, y_test)

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
                {"Binary": "Binary", "Multiclass": "Multiclass"},
                selected="Binary"
            )
        )
    ),

    # Two-column layout for Left and Right models
    ui.row(
        # Left Column
        ui.column(
            6,
            # Model selection dropdown
            ui.input_select(
                "modelSelLeft",
                "Model Selection",
                {
                    "Decision Tree": "Decision Tree",
                    "Random Forest": "Random Forest",
                    "AdaBoost": "AdaBoost",
                    "Gradient Boosting": "Gradient Boosting"
                }
            ),
            ui.h3("Model Description"),
            ui.output_text("modelDescLeft"),

            # First pair: parameter slider and tree selection with corresponding plot
            ui.input_slider(
                "paramSelLeft1",
                ui.h5("Learning Rate (alpha)"),
                min=0,
                max=1,
                value=0.5
            ),
            ui.input_select(
                "treeSelLeft1",
                ui.h5("Tree Selection"),
                {"Tree A": "Tree A", "Tree B": "Tree B"}
            ),
            ui.output_plot("treePlotLeft1"),

            # Second pair: parameter slider and tree selection with corresponding plot
            ui.input_slider(
                "paramSelLeft2",
                ui.h5("Max Depth"),
                min=0,
                max=1,
                value=0.5
            ),
            ui.input_select(
                "treeSelLeft2",
                ui.h5("Tree Selection"),
                {"Tree A": "Tree A", "Tree B": "Tree B"}
            ),
            ui.output_plot("treePlotLeft2"),

            # Results section
            ui.h4("Results (Left)"),
            ui.output_text("trainScoreLeft"),
            ui.output_text("testScoreLeft")
        ),

        # Right Column
        ui.column(
            6,
            # Model selection dropdown
            ui.input_select(
                "modelSelRight",
                "Model Selection",
                {
                    "Decision Tree": "Decision Tree",
                    "Random Forest": "Random Forest",
                    "AdaBoost": "AdaBoost",
                    "Gradient Boosting": "Gradient Boosting"
                }
            ),
            ui.h3("Model Description"),
            ui.output_text("modelDescRight"),

            # First pair: parameter slider and tree selection with corresponding plot
            ui.input_slider(
                "paramSelRight1",
                ui.h5("Learning Rate (alpha)"),
                min=0,
                max=1,
                value=0.5
            ),
            ui.input_select(
                "treeSelRight1",
                ui.h5("Tree Selection"),
                {"Tree A": "Tree A", "Tree B": "Tree B"}
            ),
            ui.output_plot("treePlotRight1"),

            # Second pair: parameter slider and tree selection with corresponding plot
            ui.input_slider(
                "paramSelRight2",
                ui.h5("Max Depth"),
                min=0,
                max=1,
                value=0.5
            ),
            ui.input_select(
                "treeSelRight2",
                ui.h5("Tree Selection"),
                {"Tree A": "Tree A", "Tree B": "Tree B"}
            ),
            ui.output_plot("treePlotRight2"),

            # Results section
            ui.h4("Results"),
            ui.output_text("trainScoreRight"),
            ui.output_text("testScoreRight")
        )
    )
)

'''
f"Selected Model: {input.modelSelLeft()}\n"
f"Classification: {input.classification_type()}\n"
f"Pair 1 - Parameter: {input.paramSelLeft1()}, Tree: {input.treeSelLeft1()}\n"
f"Pair 2 - Parameter: {input.paramSelLeft2()}, Tree: {input.treeSelLeft2()}"
'''

# Define the server logic
def server(input, output, session):
    @output
    @render.text

    def modelDescLeft():
        if input.modelSelLeft() == "Decision Tree":
            return (
                decision_tree_model.model_description
            )
        elif input.modelSelLeft() == "Random Forest":
             return (
                 random_forest_model.model_description
             )
        elif input.modelSelLeft() == "AdaBoost":
             return (
                 ada_boost_model.model_description
             )
        else:
            return (
                gradient_boosting_model.model_description
            )

    @output
    @render.text
    def modelDescRight():
        if input.modelSelRight() == "Decision Tree":
            return (
                decision_tree_model.model_description
            )
        elif input.modelSelRight() == "Random Forest":
             return (
                 random_forest_model.model_description
             )
        elif input.modelSelRight() == "AdaBoost":
             return (
                 ada_boost_model.model_description
             )
        else:
            return (
                gradient_boosting_model.model_description
            )
        '''
        return (
            f"Selected Model: {input.modelSelRight()}\n"
            f"Classification: {input.classification_type()}\n"
            f"Pair 1 - Parameter: {input.paramSelRight1()}, Tree: {input.treeSelRight1()}\n"
            f"Pair 2 - Parameter: {input.paramSelRight2()}, Tree: {input.treeSelRight2()}"
        )
        '''

    @output
    @render.plot
    def treePlotLeft1():
        if input.modelSelLeft() == "Decision Tree":
            return (
                decision_tree_model.plot_tree()
            )
        elif input.modelSelLeft() == "Random Forest":
             return (
                 random_forest_model.model_description
             )
        elif input.modelSelLeft() == "AdaBoost":
             return (
                 ada_boost_model.model_description
             )
        else:
            return (
                gradient_boosting_model.model_description
            )
        '''
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        ax.plot(np.random.rand(10))
        ax.set_title(f"Left Tree Plot 1: {input.treeSelLeft1()}")
        return fig
        '''

    @output
    @render.plot
    def treePlotLeft2():
        if input.modelSelLeft() == "Decision Tree":
            return (
                decision_tree_model.plot_tree()
            )
        elif input.modelSelLeft() == "Random Forest":
             return (
                 random_forest_model.model_description
             )
        elif input.modelSelLeft() == "AdaBoost":
             return (
                 ada_boost_model.model_description
             )
        else:
            return (
                gradient_boosting_model.model_description
            )

    @output
    @render.plot
    def treePlotRight1():
        if input.modelSelRight() == "Decision Tree":
            return (
                decision_tree_model.plot_tree()
            )
        elif input.modelSelRight() == "Random Forest":
             return (
                 random_forest_model.model_description
             )
        elif input.modelSelRight() == "AdaBoost":
             return (
                 ada_boost_model.model_description
             )
        else:
            return (
                gradient_boosting_model.model_description
            )

    @output
    @render.plot
    def treePlotRight2():
        if input.modelSelRight() == "Decision Tree":
            return (
                decision_tree_model.plot_tree()
            )
        elif input.modelSelRight() == "Random Forest":
             return (
                 random_forest_model.model_description
             )
        elif input.modelSelRight() == "AdaBoost":
             return (
                 ada_boost_model.model_description
             )
        else:
            return (
                gradient_boosting_model.model_description
            )

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
