import util as utils

def streamlit_mapper(map_value, var_name) -> int:
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
    config = utils.load_config()

    binary_mapper = {0: "no", 1:"yes"}
    mappers = {
        "weekday": config["weekday_map"],
        "season": config["season_map"],
        "weathersit": config["weathersit_map"],
        "holiday":  binary_mapper,
        "workingday": binary_mapper,
    }

    mapper = mappers[var_name]

    key_list = list(mapper.keys())
    val_list = list(mapper.values())

    pos = val_list.index(map_value)

    return key_list[pos]
