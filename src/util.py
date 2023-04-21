import yaml
import joblib
from datetime import datetime


config_dir = "config/config.yaml"

def time_stamp() -> datetime:
    """Return current date and time"""

    return datetime.now()

def load_config() -> dict: 
    """Loading configuration file"""

    try:
        with open(config_dir, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as fe:
        raise RuntimeError("Parameters file not found in path.")

    # Return params in dict format
    return config

def pickle_load(file_path: str) -> joblib.load:
    """Load and return pickle file"""

    return joblib.load(file_path)

def pickle_dump(data, file_path: str) -> None:
    """A Procedure to dump data into file"""
    joblib.dump(data, file_path)


def print_debug(messages: str) -> None:
    """A procedure to check wheter user wants to use print or not"""

    params = load_config()
    PRINT_DEBUG = params["print_debug"]

    if PRINT_DEBUG == True:
        print(time_stamp(), messages)

def streamlit_mapper(map_value, mapper) -> int:
    """This function is used to map feature to its mapper

    The output depends on the mapper. The mapper can be found
    in the configuration file.

    Parameters
    ----------
    map_value : str
        The value that should be mapped
    mapper : dict
        The dictionary of mapped key to value

    Return
    ------
    map_value : int
        The mapped value based on the pre-defined mapper
    """

    key_list = list(mapper.keys())
    val_list = list(mapper.values())

    pos = val_list.index(map_value)

    return key_list[pos]

if __name__ == "__main__":
    pass