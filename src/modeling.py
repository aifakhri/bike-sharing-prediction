import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import util as utils

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
import hashlib
import json

def load_train_feng(params: dict) -> pd.DataFrame:
    # Load train set
    x_train = utils.pickle_load(params["train_feng_set_path"][0])
    y_train = utils.pickle_load(params["train_feng_set_path"][1])

    return x_train, y_train

def load_valid_feng(params: dict) -> pd.DataFrame:
    # Load valid set
    x_valid = utils.pickle_load(params["valid_feng_set_path"][0])
    y_valid = utils.pickle_load(params["valid_feng_set_path"][1])

    return x_valid, y_valid

def load_test_feng(params: dict) -> pd.DataFrame:
    # Load tets set
    x_test = utils.pickle_load(params["test_feng_set_path"][0])
    y_test = utils.pickle_load(params["test_feng_set_path"][1])

    return x_test, y_test

def load_dataset(params: dict):
    utils.print_debug("Loading Dataset")

    # Load train set
    x_train, y_train = load_train_feng(params)

    # Load valid set
    x_valid, y_valid = load_valid_feng(params)

    # Load test set
    x_test, y_test = load_test_feng(params)

    utils.print_debug("Dataset Loaded")

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def training_log_template() -> dict:
    """This is a function to generate log template"""

    # Debug message
    utils.print_debug("Creating training log template")
    
    logger = {
        "model_name" : [],
        "model_uid" : [],
        "training_time" : [],
        "training_date" : [],
        "performance" : [],
    }

    # Debig message
    utils.print_debug("Training log template has been created")

    # Return training log template
    return logger

def training_log_updater(current_log, params) -> list:
    # Create copy of current log
    current_log = deepcopy(current_log)

    # Path for training log file
    log_path = params["training_log_path"]

    # Try to load training log file
    try:
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()

    # If file not found, create a new one
    except FileNotFoundError as fe:
        with open(log_path, "w") as file:
            file.write("[]")
        file.close()

        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    
    # Add current log to previous log
    last_log.append(current_log)

    # Save updated log
    with open(log_path, "w") as file:
        json.dump(last_log, file)
        file.close()

    # Return log
    return last_log

def create_model_object(params) -> list:
    utils.print_debug("Creating model objects")

    dtree = DecisionTreeRegressor()
    gboost = GradientBoostingRegressor()
    rforest = RandomForestRegressor()
    
    list_of_models = [
            {
                "model_name": dtree.__class__.__name__,
                "model_object": dtree,
                "model_uid": ""
            },
            {
                "model_name": rforest.__class__.__name__,
                "model_object": rforest,
                "model_uid": ""
            },
            {
                "model_name": gboost.__class__.__name__,
                "model_object": gboost,
                "model_uid": ""
            }
    ]

    utils.print_debug("Model Object has been created")

    return list_of_models

def train_eval(configuration_model: str, 
               params: dict, 
               hyperparams_model: list = None) -> tuple:

    # Load dataset
    x_train, y_train, \
    x_valid, y_valid, \
    x_test, y_test = load_dataset(params)

    # Variabel to store trained models
    list_of_trained_model = dict()

    # Create log template
    training_log = training_log_template()

    # Create model objects
    if hyperparams_model == None:
        list_of_model = create_model_object(params)
    else:
        list_of_model = deepcopy(hyperparams_model)

    # Variabel to store tained model
    trained_model = list()

    # Train each model by current dataset configuration
    for model in list_of_model:
        # Debug message
        utils.print_debug("Training model: " 
                          "{}".format(model["model_name"]))

        # Training
        training_time = utils.time_stamp()
        model["model_object"].fit(x_train, y_train)
        training_time = (utils.time_stamp() - training_time)\
                        .total_seconds()

        # Debug message
        utils.print_debug("Evalutaing model: "
                         "{}".format(model["model_name"]))

        # Evaluation
        y_predict = model["model_object"].predict(x_valid)
        performance = mean_squared_error(y_valid, y_predict)

        # Debug message
        utils.print_debug("Logging: {}".format(model["model_name"]))

        # Create UID
        uid = hashlib.md5(str(training_time).encode()).hexdigest()

        # Assign model's UID
        model["model_uid"] = uid

        # Create training log data
        training_log["model_name"].append(
            "{}-{}".format(configuration_model, model["model_name"])
        )
        training_log["model_uid"].append(uid)
        training_log["training_time"].append(training_time)
        training_log["training_date"].append(utils.time_stamp())
        training_log["performance"].append(performance)

        # Collect current trained model
        trained_model.append(deepcopy(model))

        
        # Collect current trained list of model
        list_of_trained_model = deepcopy(trained_model)
    
    # Debug message
    utils.print_debug("All combination models and "
                     "configuration data has been trained.")
    
    # Return list trained model
    return list_of_trained_model, training_log

def get_production_model(list_of_model, training_log, params):
    # Create copy list of model
    list_of_model = deepcopy(list_of_model)
    
    # Debug message
    utils.print_debug("Choosing model by metrics score.")

    # Create required predefined variabel
    curr_prod_model = None
    prev_prod_model = None
    prod_model_log = None

    # Debug message
    utils.print_debug("Converting training log type of data "
                     "from dict to dataframe.")

    # Convert dictionary to pandas for easy operation
    training_log = pd.DataFrame(deepcopy(training_log))

    # Debug message
    utils.print_debug("Trying to load previous production model.")

    # Check if there is a previous production model
    try:
        prev_prod_model = utils.pickle_load(
            params["production_model_path"]
        )
        utils.print_debug("Previous production model loaded.")
    except FileNotFoundError as fe:
        utils.print_debug("No previous production model detected, choosing" 
                         " best model only from current trained model.")

    # If previous production model detected:
    if prev_prod_model != None:
        # Debug message
        utils.print_debug("Loading validation data.")
        x_valid, y_valid = load_valid_feng(params)
        
        # Debug message
        utils.print_debug("Checking compatibilty previous production model's "
                         "input with current train data's features.")

        # Check list features of previous production model and current dataset
        prod_model_features = set(
            prev_prod_model["model_data"]["model_object"]\
            .feature_names_in_
        )
        curr_dataset_features = set(x_valid.columns)
        number_of_different_features = len(
            (prod_model_features - curr_dataset_features) \
            | (curr_dataset_features - prod_model_features))

        # If feature matched:
        if number_of_different_features == 0:
            # Debug message
            utils.print_debug("Features compatible.")

            # Debug message
            utils.print_debug("Reassesing previous model "
                             "performance using current validation data.")

            # Re-predict previous production model to provide
            # valid metrics compared to other current models
            y_pred = prev_prod_model["model_data"]["model_object"]\
                     .predict(x_valid)

            # Re-asses prediction result
            eval_res = mean_squared_error(y_valid, y_pred)

            # Debug message
            utils.print_debug("Assessing complete.")

            # Debug message
            utils.print_debug("Storing new metrics data "
                             "to previous model structure.")

            # Update their performance log
            prev_prod_model["model_log"]["performance"] = eval_res

            # Debug message
            utils.print_debug("Adding previous model data to current "
                             "training log and list of model")

            # Added previous production model log to current logs 
            # to compere who has the greatest f1 score
            training_log = pd.concat([training_log, 
                                      pd.DataFrame(
                                          [prev_prod_model["model_log"]])]
            )

            # Added previous production model to current list of models
            # to choose from if it has the greatest performance
            list_of_model.append(
                deepcopy(prev_prod_model["model_data"])
            )
        else:
            # To indicate that we are not using previous production model
            prev_prod_model = None

            # Debug message
            utils.print_debug("Different features between production " 
                              "model with current dataset is detected, "
                              "ignoring production dataset.")

    # Debug message
    utils.print_debug("Sorting training log by mean squared avg and training time.")

    # Sort training log by f1 score macro avg and trining time
    best_model_log = training_log.sort_values(
        ["performance", "training_time"], ascending = [True, True]
    ).iloc[0]
    
    # Debug message
    utils.print_debug("Searching model data based on sorted training log.")

    # Get model object with greatest f1 score macro avg by using UID
    for model_data in list_of_model:
        if model_data["model_uid"] == best_model_log["model_uid"]:
            # Create current production data
            curr_prod_model = dict()
            curr_prod_model["model_data"] = deepcopy(model_data)
            curr_prod_model["model_log"] = deepcopy(best_model_log.to_dict())

            # Formatting String
            model_name_fmt = "Production-{}".format(
                curr_prod_model["model_data"]["model_name"]
            )

            # Set training format
            training_date_fmt = str(
                curr_prod_model["model_log"]["training_date"]
            )

            # Add Formatted String to the current production model
            curr_prod_model["model_log"]["model_name"] = model_name_fmt
            curr_prod_model["model_log"]["training_date"] = training_date_fmt

            # Update Log
            prod_model_log = training_log_updater(
                curr_prod_model["model_log"], params
            )
            break
    
    # In case UID not found
    if curr_prod_model == None:
        raise RuntimeError("The best model not found in your list of model.")
    
    # Debug message
    utils.print_debug("Model chosen.")

    # Dump chosen production model
    utils.pickle_dump(curr_prod_model, params["production_model_path"])
    
    # Return current chosen production model,
    # log of production models and current training log
    return curr_prod_model, prod_model_log, training_log

def create_dist_params(model_name: str) -> dict:
    dist_params_dtree = {
        "criterion": ["squared_error", "friedman_mse"],
        "max_depth": np.arange(5, 37)
    }
    dist_params_gboost = {
        "n_estimators": [50, 100, 200, 300, 400, 500],
    }
    dist_params_rforest = {
        "n_estimators": [50, 100, 200, 300, 400, 500],
        "max_features": ["sqrt", "log2"],
        "criterion": ["squared_error", "friedman_mse"],
    }
    
    dist_params = {
        "DecisionTreeRegressor": dist_params_dtree,
        "GradientBoostingRegressor": dist_params_gboost,
        "RandomForestRegressor": dist_params_rforest
    }

    return dist_params[model_name]

def hyper_params_tuning(model: dict) -> list:
    model = deepcopy(model)
    
    model_name = model["model_data"]["model_name"]
    model_object = model["model_data"]["model_object"]
    
    dist_params = create_dist_params(model_name)
    model_rsc = GridSearchCV(estimator=model_object,
                             param_grid=dist_params,
                             n_jobs=-1,
                             cv=10)

    model_data = {
        "model_name": model_name,
        "model_object": model_rsc,
        "model_uid": ""
    }
    
    return [model_data]

if __name__ == "__main__":
    #1. Load configuration file
    params = utils.load_config()


    # 2. Train and evaluate model
    list_of_trained_model, training_logs = train_eval(
        configuration_model="baseline",
        params=params
    )

    # 3. Choose the best model for production
    model, production_model_log, training_logs = get_production_model(
        list_of_model=list_of_trained_model,
        training_log=training_logs,
        params=params
    )

    # 4. Optimizing with hyperparams tuning
    list_of_trained_model, training_log = train_eval(
        configuration_model="Hyperparameter_Tuning",
        params=params,
        hyperparams_model=hyper_params_tuning(model)
    )

    # 5. Choose the best model for production
    model, production_model_log, training_logs = get_production_model(
        list_of_model=list_of_trained_model,
        training_log=training_logs,
        params=params
    )