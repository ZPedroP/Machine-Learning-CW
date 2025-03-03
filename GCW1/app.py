# TODO: Currently the plot_tree() function is being used to plot and get the results. Right now it is resulting in this function being called twice. As long as the random state is kept the same, the accuracy results should represent the results from the tree correctly. The code should be update so it only calls the plot_tree() function once.

import os
import sys

ROOT_PATH = os.path.dirname(__file__)

from shiny import ui, render, App
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
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
                },
                selected="Decision Tree"
            ),
            ui.h3("Model Description"),
            ui.output_text("modelDescLeft"),

            # Parameter slider and tree selection with corresponding plot
            ui.panel_conditional(
                "input.modelSelLeft === 'Decision Tree'",
                ui.input_slider(
                    "paramSelLeftDt1",
                    ui.h5("Cost Complexity Pruning-α"),
                    min=0,
                    max=0.05,
                    value=0.01,
                    sep=0.005
                ),
                ui.input_slider(
                    "paramSelLeftDt2",
                    ui.h5("Max Depth"),
                    min=1,
                    max=20,
                    value=10,
                    step=1
                ),
            ),

            ui.panel_conditional(
                "input.modelSelLeft === 'Random Forest'",
                ui.row(
                    ui.column(
                        6,
                        ui.input_slider(
                            "paramSelLeftRf1",
                            ui.h5("Cost Complexity Pruning-α"),
                            min=0,
                            max=0.05,
                            value=0.01,
                            sep=0.005
                        ),
                        ui.input_slider(
                            "paramSelLeftRf2",
                            ui.h5("Max Depth"),
                            min=1,
                            max=20,
                            value=10,
                            step=1
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_slider(
                            "paramSelLeftRf3",
                            ui.h5("Number of Estimators"),
                            min=1,
                            max=100,
                            value=50,
                            step=1
                        ),
                        ui.input_slider(
                            "paramSelLeftRf4",
                            ui.h5("Max Features"),
                            min=1,
                            max=60,
                            value=30,
                            step=1
                        ),
                    ),
                ),
                # Dynamically select the tree to plot
                ui.output_ui("dynamic_tree_selection")
            ),

            # ui.panel_conditional(
            #     "input.modelSelLeft != 'Decision Tree'",
            #     ui.input_select(
            #         "treeSelLeft1",
            #         ui.h5("Tree Selection"),
            #         {"Tree A": "Tree A", "Tree B": "Tree B"}
            #     )
            # ),

            ui.output_plot("treePlotLeft1"),

            # Results section
            ui.h4("Results"),
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
                },
                selected="Random Forest"
            ),
            ui.h3("Model Description"),
            ui.output_text("modelDescRight"),

            # Parameter slider and tree selection with corresponding plot
            ui.panel_conditional(
                "input.modelSelRight === 'Decision Tree'",
                ui.input_slider(
                    "paramSelRight1",
                    ui.h5("Cost Complexity Pruning-α"),
                    min=0,
                    max=0.05,
                    value=0.01,
                    sep=0.005
                ),
            ),

            ui.input_slider(
                "paramSelRight2",
                ui.h5("Max Depth"),
                min=1,
                max=20,
                value=10,
                step=1
            ),

            ui.panel_conditional(
                "input.modelSelRight != 'Decision Tree'",
                ui.input_select(
                    "treeSelRight1",
                    ui.h5("Tree Selection"),
                    {"Tree A": "Tree A", "Tree B": "Tree B"}
                )
            ),

            ui.output_plot("treePlotRight1"),

            # Results section
            ui.h4("Results"),
            ui.output_text("trainScoreRight"),
            ui.output_text("testScoreRight")
        )
    )
)

# Define the server logic
def server(input, output, session):
    @output
    @render.ui
    def dynamic_tree_selection():
        if input.modelSelLeft() == "Random Forest":
            num_estimators = input.paramSelLeftRf3()  # Get the selected number of estimators
            options = {str(i): f"Tree {i+1}" for i in range(num_estimators)}
            # options = {f"Tree {i+1}": f"Tree {i+1}" for i in range(num_estimators)}
            return ui.input_select("treeSelLeftRf", ui.h5("Tree Selection"), options)
        return None

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

    @output
    @render.plot
    def treePlotLeft1():
        if input.modelSelLeft() == "Decision Tree":
            return (
                decision_tree_model.plot_results(ccp_alpha=input.paramSelLeftDt1(), max_depth=input.paramSelLeftDt2())
            )
        elif input.modelSelLeft() == "Random Forest":
             rf_left_estimators = random_forest_model.plot_results(ccp_alpha=input.paramSelLeftRf1() , max_depth=input.paramSelLeftRf2(), n_estimators=input.paramSelLeftRf3(), max_features=input.paramSelLeftRf4())
             fig, ax = plt.subplots()
             plot_tree(rf_left_estimators[int(input.treeSelLeftRf())], feature_names=random_forest_model.X_train.columns, class_names=['0', '1'], filled=True, ax=ax)
             ax.set_title(f"Random Forest")
             return (
                fig
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
                decision_tree_model.plot_results(ccp_alpha=input.paramSelRight1())
            )
        elif input.modelSelRight() == "Random Forest":
            #  rf_right = random_forest_model.plot_tree(n_estimators=input.paramSelRight1(), max_depth=input.paramSelRight2())
            #  fig, ax = plt.subplots()
            #  plot_tree(rf_right, feature_names=random_forest_model.X_train.columns, class_names=['0', '1'], filled=True, ax=ax)
            #  ax.set_title(f"Random Forest (number of estimators={input.paramSelRight1()})")
            #  return (
            #     fig
            #  )
            ada_boost_model.model_description
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
        if input.modelSelLeft() == "Decision Tree":
            return (
                "Train Score: {:.4}".format(decision_tree_model.plot_results(ccp_alpha=input.paramSelLeftDt1(), max_depth=input.paramSelLeftDt2(), return_results=1)[0])
            )
        elif input.modelSelLeft() == "Random Forest":
             return (
                 "Train Score (Right): 0.92"
             )
        elif input.modelSelLeft() == "AdaBoost":
             return (
                 "Train Score (Right): 0.92"
             )
        else:
            return (
                "Train Score (Right): 0.92"
            )

    @output
    @render.text
    def testScoreLeft():
        if input.modelSelLeft() == "Decision Tree":
            return (
                "Test Score: {:.4}".format(decision_tree_model.plot_results(ccp_alpha=input.paramSelLeftDt1(), max_depth=input.paramSelLeftDt2(), return_results=1)[1])
            )
        elif input.modelSelLeft() == "Random Forest":
             return (
                 "Test Score (Left): 0.90"
             )
        elif input.modelSelLeft() == "AdaBoost":
             return (
                 "Test Score (Left): 0.90"
             )
        else:
            return (
                "Test Score (Left): 0.90"
            )

    @output
    @render.text
    def trainScoreRight():
        if input.modelSelRight() == "Decision Tree":
            return (
                "Train Score: {:.4}".format(decision_tree_model.plot_results(ccp_alpha=input.paramSelRight1(), max_depth=input.paramSelRight2(), return_results=1)[0])
            )
        elif input.modelSelRight() == "Random Forest":
             return (
                 "Train Score (Right): 0.92"
             )
        elif input.modelSelRight() == "AdaBoost":
             return (
                 "Train Score (Right): 0.92"
             )
        else:
            return (
                "Train Score (Right): 0.92"
            )

    @output
    @render.text
    def testScoreRight():
        if input.modelSelRight() == "Decision Tree":
            return (
                "Train Score: {:.4}".format(decision_tree_model.plot_results(ccp_alpha=input.paramSelRight1(), max_depth=input.paramSelRight2(), return_results=1)[1])
            )
        elif input.modelSelRight() == "Random Forest":
             return (
                 "Test Score (Right): 0.88"
             )
        elif input.modelSelRight() == "AdaBoost":
             return (
                 "Test Score (Right): 0.88"
             )
        else:
            return (
                "Test Score (Right): 0.88"
            )

# Create the Shiny app object
app = App(app_ui, server)
