import os
import pandas as pd
import util as utils

from tqdm import tqdm
from copy import deepcopy

from sklearn.model_selection import train_test_split



def read_raw_data(config_data: dict) -> pd.DataFrame:
    """Read the raw data frame

    Parameters
    ----------
    config : dict
        The loaded configuration file

    Return
    ------
    raw_dataset : pd.DataFrame
        the loaded dataframe in pandas DataFrame
    """

    # Define dataset directory
    raw_dataset_dir = config_data["raw_data_set_dir"]

    # Get the data in CSV
    for flname in tqdm(os.listdir(raw_dataset_dir)):
        if flname.endswith(".csv"):
            raw_dataset = pd.read_csv(raw_dataset_dir + flname)

    return raw_dataset

def basic_checking(data) -> None:
    """Checking Null Values and Duplicates

    data : pd.DataFrame
        The dataset we want to check
    """

    # Check Null Values
    n_null = data.isnull().any().sum()

    if n_null > 0:
        print("There are missing values, Please check it")
    else:
        print("There are no missing values")

    # Check Duplicates
    n_duplicates = data.duplicated().any().sum()

    if n_duplicates > 0:
        print("There are duplicates, Please check it")
    else:
        print("There are no duplicates")

def remove_features(features: list, data: pd.DataFrame) -> pd.DataFrame:
    """Removing various features on the dataset

    Parameters
    ----------
    features : list
        the list of feature that should be deleted
    data : pd.DataFrame
        The dataset of which the feature resides

    Return
    ------
    data : pd.DataFrame
        The dataset withouth the unnecessary features
    """

    if isinstance(features, list):
        data = data.drop(features, axis=1, inplace=True)
        return data
    else:
        fail_msg = "Please Enter a list"
        raise fail_msg

def reformat_feature(feature, data):
    """Procedure to reformat the data point in the dataset

    Currently the function is only supporting data formating
    in a sring date form: YY-mm-dd, and then get the date value

    Parameters
    ----------
    feature : str
        The name of feature we want to format
    data : pd.DataFrame
        The dataset of which the feature resides

    Return
    ------
    data : pd.DataFrame
        The dataset withouth the unnecessary features
    """

    # Get the day data from date
    temp_dict = {}
    for i in data[feature].unique().tolist():
        try:
            temp_dict[i] = int(i.split("-")[-1].lstrip("0"))
        except ValueError:
            print("Data is not integer")
            exit()

    data[feature].replace(temp_dict, inplace=True)

def validate_data(data, config_data):
    """
    """

    for feature in data:
        try:
            if feature == config_data["label"]:
                continue
            elif feature in config_data["categorical_columns"]:
                assert set(data[feature]).issubset(
                    set(config_data[f"range_{feature}"])), "error occurs"
            else:
                assert data[feature].between(
                    config_data[f"range_{feature}"][0],
                    config_data[f"range_{feature}"][1]) \
                    .sum() == len(data), "error occurs"
        except KeyError:
            print("Variable Has Been Dropped")
    print("Data is Valid")

if __name__ == "__main__":
    # 1. Load Configuration File
    config_data = utils.load_config()

    # 2. Load Dataset
    raw_dataset = read_raw_data(config_data=config_data)

    # 3. Save Dataset
    utils.pickle_dump(data=raw_dataset,
                      file_path=config_data["raw_data_set_path"])

    # 4. Simple Checking
    basic_checking(data=raw_dataset)

    # 4. Dropping Unnecessary Features
    feature_list = config_data["unnecessary_predictors"]
    remove_features(features=feature_list, data=raw_dataset)

    # 5. Formatting 'dteday' feature
    reformat_feature(feature="dteday", data=raw_dataset)

    # 6. Validate Data
    validate_data(data=raw_dataset, config_data=config_data)

    # 7. Split Data
    # 7.1 Splitting Input Output
    x = raw_dataset[config_data["predictors"]].copy()
    y = raw_dataset[config_data["label"]].copy()

    # 7.1 Split Train Test
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.3,
                                                        random_state=123)

    # 7.2 Splitting Test Valid
    x_valid, x_test, y_valid, y_test = train_test_split(x_test,
                                                        y_test,
                                                        test_size=0.5,
                                                        random_state=123)

    # 8. Save train, valid and test set into pickle
    # Dump train data
    utils.pickle_dump(x_train, config_data["train_set_path"][0])
    utils.pickle_dump(y_train, config_data["train_set_path"][1])

    # Dump validation data
    utils.pickle_dump(x_valid, config_data["valid_set_path"][0])
    utils.pickle_dump(x_valid, config_data["valid_set_path"][1])

    # Dump test data
    utils.pickle_dump(x_test, config_data["test_set_path"][0])
    utils.pickle_dump(x_test, config_data["test_set_path"][1])
