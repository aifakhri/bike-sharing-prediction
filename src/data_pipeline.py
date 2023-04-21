import os
import pandas as pd
import util as utils

from tqdm import tqdm
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

    # Create Pandas Object
    raw_dataset = pd.DataFrame()

    # Define dataset directory
    raw_dataset_dir = config["dataset_path"]

    # Get the data in CSV
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat(pd.read_csv(raw_dataset_dir + i),
                                raw_dataset)

    return raw_dataset