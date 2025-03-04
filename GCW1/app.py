# TODO: Currently the plot_tree() function is being used to plot and get the results. Right now it is resulting in this function being called twice. As long as the random state is kept the same, the accuracy results should represent the results from the tree correctly. The code should be update so it only calls the plot_tree() function once.

import os
import sys

ROOT_PATH = os.path.dirname(__file__)

from shiny import ui, render, reactive, App
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
# data_processor = DataProcessor(file_path)
# X, y = data_processor.load_data()
# X_preprocessed_df = data_processor.preprocess(X)
# X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_df, y, test_size=0.3, random_state=42)

# decision_tree_model = DecisionTreeModel(X_train, y_train, X_test, y_test)
# random_forest_model = RandomForestModel(X_train, y_train, X_test, y_test)
# ada_boost_model = AdaBoostModel(X_train, y_train, X_test, y_test)
# gradient_boosting_model = GradientBoostingModel(X_train, y_train, X_test, y_test)

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
                    step=0.005
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
                            step=0.005
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
                ui.output_ui("dynamic_tree_selection_left_rf")
            ),

            ui.panel_conditional(
                "input.modelSelLeft === 'AdaBoost'",
                ui.row(
                    ui.column(
                        6,
                        ui.input_slider(
                            "paramSelLeftAda1",
                            ui.h5("Learning Rate"),
                            min=0,
                            max=1,
                            value=0.5,
                            step=0.01
                        ),
                        ui.input_slider(
                            "paramSelLeftAda2",
                            ui.h5("Base Estimator Max Depth"),
                            min=1,
                            max=20,
                            value=10,
                            step=1
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_slider(
                            "paramSelLeftAda3",
                            ui.h5("Number of Estimators"),
                            min=1,
                            max=100,
                            value=50,
                            step=1
                        ),
                    ),
                ),
                # Dynamically select the tree to plot
                ui.output_ui("dynamic_tree_selection_left_ada")
            ),

            ui.panel_conditional(
                "input.modelSelLeft === 'Gradient Boosting'",
                ui.row(
                    ui.column(
                        6,
                        ui.input_slider(
                            "paramSelLeftGb1",
                            ui.h5("Learning Rate"),
                            min=0,
                            max=1,
                            value=0.5,
                            step=0.01
                        ),
                        ui.input_slider(
                            "paramSelLeftGb2",
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
                            "paramSelLeftGb3",
                            ui.h5("Number of Estimators"),
                            min=1,
                            max=100,
                            value=50,
                            step=1
                        ),
                        ui.input_slider(
                            "paramSelLeftGb4",
                            ui.h5("Max Features"),
                            min=1,
                            max=60,
                            value=30,
                            step=1
                        ),
                    ),
                ),
                # Dynamically select the tree to plot
                ui.output_ui("dynamic_tree_selection_left_gb")
            ),

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
                    "paramSelRightDt1",
                    ui.h5("Cost Complexity Pruning-α"),
                    min=0,
                    max=0.05,
                    value=0.01,
                    step=0.005
                ),
                ui.input_slider(
                    "paramSelRightDt2",
                    ui.h5("Max Depth"),
                    min=1,
                    max=20,
                    value=10,
                    step=1
                ),
            ),

            ui.panel_conditional(
                "input.modelSelRight === 'Random Forest'",
                ui.row(
                    ui.column(
                        6,
                        ui.input_slider(
                            "paramSelRightRf1",
                            ui.h5("Cost Complexity Pruning-α"),
                            min=0,
                            max=0.05,
                            value=0.01,
                            step=0.005
                        ),
                        ui.input_slider(
                            "paramSelRightRf2",
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
                            "paramSelRightRf3",
                            ui.h5("Number of Estimators"),
                            min=1,
                            max=100,
                            value=50,
                            step=1
                        ),
                        ui.input_slider(
                            "paramSelRightRf4",
                            ui.h5("Max Features"),
                            min=1,
                            max=60,
                            value=30,
                            step=1
                        ),
                    ),
                ),
                # Dynamically select the tree to plot
                ui.output_ui("dynamic_tree_selection_right_rf")
            ),

            ui.panel_conditional(
                "input.modelSelRight === 'AdaBoost'",
                ui.row(
                    ui.column(
                        6,
                        ui.input_slider(
                            "paramSelRightAda1",
                            ui.h5("Learning Rate"),
                            min=0,
                            max=1,
                            value=0.5,
                            step=0.01
                        ),
                        ui.input_slider(
                            "paramSelRightAda2",
                            ui.h5("Base Estimator Max Depth"),
                            min=1,
                            max=20,
                            value=10,
                            step=1
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_slider(
                            "paramSelRightAda3",
                            ui.h5("Number of Estimators"),
                            min=1,
                            max=100,
                            value=50,
                            step=1
                        ),
                    ),
                ),
                # Dynamically select the tree to plot
                ui.output_ui("dynamic_tree_selection_right_ada")
            ),

            ui.panel_conditional(
                "input.modelSelRight === 'Gradient Boosting'",
                ui.row(
                    ui.column(
                        6,
                        ui.input_slider(
                            "paramSelRightGb1",
                            ui.h5("Learning Rate"),
                            min=0,
                            max=1,
                            value=0.5,
                            step=0.01
                        ),
                        ui.input_slider(
                            "paramSelRightGb2",
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
                            "paramSelRightGb3",
                            ui.h5("Number of Estimators"),
                            min=1,
                            max=100,
                            value=50,
                            step=1
                        ),
                        ui.input_slider(
                            "paramSelRightGb4",
                            ui.h5("Max Features"),
                            min=1,
                            max=60,
                            value=30,
                            step=1
                        ),
                    ),
                ),
                # Dynamically select the tree to plot
                ui.output_ui("dynamic_tree_selection_right_gb")
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
    # Reactive storage for models
    decision_tree_model = reactive.value(None)
    random_forest_model = reactive.value(None)
    ada_boost_model = reactive.value(None)
    gradient_boosting_model = reactive.value(None)
    data_processor = reactive.value(None)

    # Function to update models based on classification type
    def update_models():
        multiclass_option = 1 if input.classification_type() == "Multiclass" else 0
        # data_processor, X, y, X_train, X_test, y_train, y_test

        data_processor.set(DataProcessor(file_path, multiclass=multiclass_option))
        X, y = data_processor().load_data()
        X_preprocessed_df = data_processor().preprocess(X)
        X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_df, y, test_size=0.3, random_state=42)

        # Update models reactively
        # global decision_tree_model, random_forest_model, ada_boost_model, gradient_boosting_model
        decision_tree_model.set(DecisionTreeModel(X_train, y_train, X_test, y_test))
        random_forest_model.set(RandomForestModel(X_train, y_train, X_test, y_test))
        ada_boost_model.set(AdaBoostModel(X_train, y_train, X_test, y_test))
        gradient_boosting_model.set(GradientBoostingModel(X_train, y_train, X_test, y_test))

        # **Mark data as updated**
        # data_updated.set(True)

    # **Trigger update when classification_type changes**
    @reactive.effect
    def watch_classification_type():
        update_models()

    # Ensure all outputs depend on the new dataset when `classification_type` changes
    # @reactive.effect
    # def ensure_dependency():
    #     data_updated.get()  # Forces reactive dependency

    @output
    @render.ui
    def dynamic_tree_selection_left_rf():
        if input.modelSelLeft() == "Random Forest":
            num_estimators = input.paramSelLeftRf3()  # Get the selected number of estimators
            options = {str(i): f"Tree {i+1}" for i in range(num_estimators)}
            return ui.input_select("treeSelLeftRf", ui.h5("Tree Selection"), options)
        return None

    @output
    @render.ui
    def dynamic_tree_selection_right_rf():
        if input.modelSelRight() == "Random Forest":
            num_estimators = input.paramSelRightRf3()  # Get the selected number of estimators
            options = {str(i): f"Tree {i+1}" for i in range(num_estimators)}
            return ui.input_select("treeSelRightRf", ui.h5("Tree Selection"), options)
        return None
    
    @output
    @render.ui
    def dynamic_tree_selection_left_ada():
        if input.modelSelLeft() == "AdaBoost":
            num_estimators = input.paramSelLeftAda3()  # Get the selected number of estimators
            options = {str(i): f"Tree {i+1}" for i in range(num_estimators)}
            return ui.input_select("treeSelLeftAda", ui.h5("Tree Selection"), options)
        return None

    @output
    @render.ui
    def dynamic_tree_selection_right_ada():
        if input.modelSelRight() == "AdaBoost":
            num_estimators = input.paramSelRightAda3()  # Get the selected number of estimators
            options = {str(i): f"Tree {i+1}" for i in range(num_estimators)}
            return ui.input_select("treeSelRightAda", ui.h5("Tree Selection"), options)
        return None
    
    @output
    @render.ui
    def dynamic_tree_selection_left_gb():
        if input.modelSelLeft() == "Gradient Boosting":
            num_estimators = input.paramSelLeftGb3()  # Get the selected number of estimators
            options = {str(i): f"Tree {i+1}" for i in range(num_estimators)}
            return ui.input_select("treeSelLeftGb", ui.h5("Tree Selection"), options)
        return None

    @output
    @render.ui
    def dynamic_tree_selection_right_gb():
        if input.modelSelRight() == "Gradient Boosting":
            num_estimators = input.paramSelRightGb3()  # Get the selected number of estimators
            options = {str(i): f"Tree {i+1}" for i in range(num_estimators)}
            return ui.input_select("treeSelRightGb", ui.h5("Tree Selection"), options)
        return None

    @output
    @render.text
    def modelDescLeft():
        if input.modelSelLeft() == "Decision Tree":
            return (
                decision_tree_model().model_description
            )
        elif input.modelSelLeft() == "Random Forest":
             return (
                 random_forest_model().model_description
             )
        elif input.modelSelLeft() == "AdaBoost":
             return (
                 ada_boost_model().model_description
             )
        else:
            return (
                gradient_boosting_model().model_description
            )

    @output
    @render.text
    def modelDescRight():
        if input.modelSelRight() == "Decision Tree":
            return (
                decision_tree_model().model_description
            )
        elif input.modelSelRight() == "Random Forest":
             return (
                 random_forest_model().model_description
             )
        elif input.modelSelRight() == "AdaBoost":
             return (
                 ada_boost_model().model_description
             )
        else:
            return (
                gradient_boosting_model().model_description
            )

    @output
    @render.plot
    def treePlotLeft1():
        if input.modelSelLeft() == "Decision Tree":
            return (
                decision_tree_model().plot_results(ccp_alpha=input.paramSelLeftDt1(), max_depth=input.paramSelLeftDt2(), class_names=data_processor().class_names)
            )
        elif input.modelSelLeft() == "Random Forest":
            rf_left_estimators = random_forest_model().plot_results(ccp_alpha=input.paramSelLeftRf1(), max_depth=input.paramSelLeftRf2(), n_estimators=input.paramSelLeftRf3(), max_features=input.paramSelLeftRf4())
            fig, ax = plt.subplots()
            plot_tree(rf_left_estimators[int(input.treeSelLeftRf())], feature_names=random_forest_model().X_train.columns, class_names=data_processor().class_names, filled=True, ax=ax)
            ax.set_title(f"Random Forest")
            return (
                fig
            )
        elif input.modelSelLeft() == "AdaBoost":
            ada_left_estimators = ada_boost_model().plot_results(learning_rate=input.paramSelLeftAda1(), base_estimator_max_depth=input.paramSelLeftAda2(), n_estimators=input.paramSelLeftAda3())
            fig, ax = plt.subplots()
            plot_tree(ada_left_estimators[int(input.treeSelLeftAda())], feature_names=ada_boost_model().X_train.columns, class_names=data_processor().class_names, filled=True, ax=ax)
            ax.set_title(f"AdaBoost")
            return (
                fig
            )
        else:
            gb_left_estimators = gradient_boosting_model().plot_results(learning_rate=input.paramSelLeftGb1(), max_depth=input.paramSelLeftGb2(), n_estimators=input.paramSelLeftGb3(), max_features=input.paramSelLeftGb4())
            fig, ax = plt.subplots()
            plot_tree(gb_left_estimators[int(input.treeSelLeftGb())][0], feature_names=gradient_boosting_model().X_train.columns, class_names=data_processor().class_names, filled=True, ax=ax)
            ax.set_title(f"Gradient Boosting")
            return (
                fig
            )

    @output
    @render.plot
    def treePlotRight1():
        if input.modelSelRight() == "Decision Tree":
            return (
                decision_tree_model().plot_results(ccp_alpha=input.paramSelRightDt1(), max_depth=input.paramSelRightDt2(), class_names=data_processor().class_names)
            )
        elif input.modelSelRight() == "Random Forest":
            rf_right_estimators = random_forest_model().plot_results(ccp_alpha=input.paramSelRightRf1(), max_depth=input.paramSelRightRf2(), n_estimators=input.paramSelRightRf3(), max_features=input.paramSelRightRf4())
            fig, ax = plt.subplots()
            plot_tree(rf_right_estimators[int(input.treeSelRightRf())], feature_names=random_forest_model().X_train.columns, class_names=data_processor().class_names, filled=True, ax=ax)
            ax.set_title(f"Random Forest")
            return (
                fig
            )
        elif input.modelSelRight() == "AdaBoost":
            ada_right_estimators = ada_boost_model().plot_results(learning_rate=input.paramSelRightAda1(), base_estimator_max_depth=input.paramSelRightAda2(), n_estimators=input.paramSelRightAda3())
            fig, ax = plt.subplots()
            plot_tree(ada_right_estimators[int(input.treeSelRightAda())], feature_names=ada_boost_model().X_train.columns, class_names=data_processor().class_names, filled=True, ax=ax)
            ax.set_title(f"AdaBoost")
            return (
                fig
            )
        else:
            gb_right_estimators = gradient_boosting_model().plot_results(learning_rate=input.paramSelRightGb1(), max_depth=input.paramSelRightGb2(), n_estimators=input.paramSelRightGb3(), max_features=input.paramSelRightGb4())
            fig, ax = plt.subplots()
            plot_tree(gb_right_estimators[int(input.treeSelRightGb())][0], feature_names=gradient_boosting_model().X_train.columns, class_names=data_processor().class_names, filled=True, ax=ax)
            ax.set_title(f"Gradient Boosting")
            return (
                fig
            )

    @output
    @render.text
    def trainScoreLeft():
        if input.modelSelLeft() == "Decision Tree":
            return (
                "Train Score: {:.4}".format(decision_tree_model().plot_results(ccp_alpha=input.paramSelLeftDt1(), max_depth=input.paramSelLeftDt2(), return_results=1, class_names=data_processor().class_names)[0])
            )
        elif input.modelSelLeft() == "Random Forest":
             return (
                "Train Score: {:.4}".format(random_forest_model().plot_results(ccp_alpha=input.paramSelLeftRf1() , max_depth=input.paramSelLeftRf2(), n_estimators=input.paramSelLeftRf3(), max_features=input.paramSelLeftRf4(), return_results=1)[0])
             )
        elif input.modelSelLeft() == "AdaBoost":
             return (
                "Train Score: {:.4}".format(ada_boost_model().plot_results(learning_rate=input.paramSelLeftAda1() , base_estimator_max_depth=input.paramSelLeftAda2(), n_estimators=input.paramSelLeftAda3(), return_results=1)[0])
             )
        else:
            return (
                "Train Score: {:.4}".format(gradient_boosting_model().plot_results(learning_rate=input.paramSelLeftGb1(), max_depth=input.paramSelLeftGb2(), n_estimators=input.paramSelLeftGb3(), max_features=input.paramSelLeftGb4(), return_results=1)[0])
            )

    @output
    @render.text
    def testScoreLeft():
        if input.modelSelLeft() == "Decision Tree":
            return (
                "Test Score: {:.4}".format(decision_tree_model().plot_results(ccp_alpha=input.paramSelLeftDt1(), max_depth=input.paramSelLeftDt2(), return_results=1, class_names=data_processor().class_names)[1])
            )
        elif input.modelSelLeft() == "Random Forest":
             return (
                "Test Score: {:.4}".format(random_forest_model().plot_results(ccp_alpha=input.paramSelLeftRf1() , max_depth=input.paramSelLeftRf2(), n_estimators=input.paramSelLeftRf3(), max_features=input.paramSelLeftRf4(), return_results=1)[1])
             )
        elif input.modelSelLeft() == "AdaBoost":
             return (
                "Test Score: {:.4}".format(ada_boost_model().plot_results(learning_rate=input.paramSelLeftAda1() , base_estimator_max_depth=input.paramSelLeftAda2(), n_estimators=input.paramSelLeftAda3(), return_results=1)[1])
             )
        else:
            return (
                "Test Score: {:.4}".format(gradient_boosting_model().plot_results(learning_rate=input.paramSelLeftGb1(), max_depth=input.paramSelLeftGb2(), n_estimators=input.paramSelLeftGb3(), max_features=input.paramSelLeftGb4(), return_results=1)[1])
            )

    @output
    @render.text
    def trainScoreRight():
        if input.modelSelRight() == "Decision Tree":
            return (
                "Train Score: {:.4}".format(decision_tree_model().plot_results(ccp_alpha=input.paramSelRightDt1(), max_depth=input.paramSelRightDt2(), return_results=1, class_names=data_processor().class_names)[0])
            )
        elif input.modelSelRight() == "Random Forest":
             return (
                "Train Score: {:.4}".format(random_forest_model().plot_results(ccp_alpha=input.paramSelRightRf1() , max_depth=input.paramSelRightRf2(), n_estimators=input.paramSelRightRf3(), max_features=input.paramSelRightRf4(), return_results=1)[0])
             )
        elif input.modelSelRight() == "AdaBoost":
             return (
                "Train Score: {:.4}".format(ada_boost_model().plot_results(learning_rate=input.paramSelRightAda1() , base_estimator_max_depth=input.paramSelRightAda2(), n_estimators=input.paramSelRightAda3(), return_results=1)[0])
             )
        else:
            return (
                "Train Score: {:.4}".format(gradient_boosting_model().plot_results(learning_rate=input.paramSelRightGb1(), max_depth=input.paramSelRightGb2(), n_estimators=input.paramSelRightGb3(), max_features=input.paramSelRightGb4(), return_results=1)[0])
            )

    @output
    @render.text
    def testScoreRight():
        if input.modelSelRight() == "Decision Tree":
            return (
                "Test Score: {:.4}".format(decision_tree_model().plot_results(ccp_alpha=input.paramSelRightDt1(), max_depth=input.paramSelRightDt2(), return_results=1, class_names=data_processor().class_names)[1])
            )
        elif input.modelSelRight() == "Random Forest":
             return (
                "Train Score: {:.4}".format(random_forest_model().plot_results(ccp_alpha=input.paramSelRightRf1() , max_depth=input.paramSelRightRf2(), n_estimators=input.paramSelRightRf3(), max_features=input.paramSelRightRf4(), return_results=1)[1])
             )
        elif input.modelSelRight() == "AdaBoost":
             return (
                "Test Score: {:.4}".format(ada_boost_model().plot_results(learning_rate=input.paramSelRightAda1() , base_estimator_max_depth=input.paramSelRightAda2(), n_estimators=input.paramSelRightAda3(), return_results=1)[1])
             )
        else:
            return (
                "Test Score: {:.4}".format(gradient_boosting_model().plot_results(learning_rate=input.paramSelRightGb1(), max_depth=input.paramSelRightGb2(), n_estimators=input.paramSelRightGb3(), max_features=input.paramSelRightGb4(), return_results=1)[1])
            )

# Create the Shiny app object
app = App(app_ui, server)
