import os
import pandas as pd
import util as utils

from tqdm import tqdm
from copy import deepcopy

from sklearn.model_selection import train_test_split



def read_raw_data(config: dict) -> pd.DataFrame:
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
    raw_dataset_dir = config["dataset_path"]

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

def check_data(input_data, params, api=False):
    pass
