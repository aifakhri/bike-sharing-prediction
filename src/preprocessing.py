import pandas as pd
import numpy as np
import util as utils

from sklearn.preprocessing import OneHotEncoder

def load_dataset(config_data: dict) -> tuple:
    """Function to load the dataset in pickle to Pandas dataframe

    The return value always in the following order: test, valid and test

    Params
    ------
    config_data : dict
        loaded YAML configuration file

    Return
    ------
    train_set, valid_set, test_set : tuple
        These the concatenated train data, validation data and test data
        in Pndas DataFrame
    """

    # loading train data
    x_train = utils.pickle_load(config_data["train_set_path"][0])
    y_train = utils.pickle_load(config_data["train_set_path"][1])

    # loading validation data
    x_valid = utils.pickle_load(config_data["valid_set_path"][0])
    y_valid = utils.pickle_load(config_data["valid_set_path"][1])

    # loading test data
    x_test = utils.pickle_load(config_data["test_set_path"][0])
    y_test = utils.pickle_load(config_data["test_set_path"][1])

    # concatenate train, valid and test data
    train_set = pd.concat([x_train, y_train], axis=1)
    valid_set = pd.concat([x_valid, y_valid], axis=1)
    test_set = pd.concat([x_test, y_test], axis=1)

    return train_set, valid_set, test_set

def remove_outliers(set_data: pd.DataFrame) -> pd.DataFrame:
    """Function to remove outliers from data

    Params
    ------
    set_data : pd.DataFrame
        the concatenated dataset, i.e x_train + y_train

    return
    ------
    cleaned_data : pd.DataFrame
        dataset without outliers
    """
    set_data = set_data.copy()
    list_of_set_data = list()

    for col_name in set_data.columns[:-1]:
        q1 = set_data[col_name].quantile(0.25)
        q3 = set_data[col_name].quantile(0.75)
        iqr = q3 - q1

        lower_q = set_data[col_name] < q1 - 1.5 * iqr
        upper_q = set_data[col_name] > q3 + 1.5 * iqr
 
        set_data_cleaned = set_data[~( lower_q | upper_q )].copy()
        list_of_set_data.append(set_data_cleaned.copy())
    
    set_data_cleaned = pd.concat(list_of_set_data)
    n_duplicated_index = set_data_cleaned.index.value_counts()


    used_index_data = n_duplicated_index[
        n_duplicated_index == (set_data.shape[1]-1)].index

    set_data_cleaned = set_data_cleaned.loc[used_index_data].drop_duplicates()

    return set_data_cleaned

def ohe_fit(feature, data_range) -> OneHotEncoder.fit:
    """ This is a function to fit OHE data
    """

    # Build and fit the OHE object
    ohe_model = OneHotEncoder(sparse_output=False)
    ohe_model.fit(np.array(data_range).reshape(-1, 1))

    # Save OHE object
    dump_filename = f"models/ohe_{feature}.pkl"
    utils.pickle_dump(ohe_model, dump_filename)

    return ohe_model
    

def ohe_transform(feature, data, ohe_model) -> pd.DataFrame:
    """Creating Ohe Model and Load it to the confguration file

    Params
    ------
    feature : str
        name of the feature we would encode
    
    config : dict
        loaded yaml configuration file

    ohe_model : OneHotEncoder
        Fitted Ohe for a specific Variable
        
    Return
    ------
    data : pd.DataFrame
        actual data with the transformed feature with OHE
    """

    # Transforming Data
    range_feature = f"range_{feature}"

    # Transforming feature with fitted OHE
    transform_feature = ohe_model.transform(
        np.array(data[feature].to_list()).reshape(-1, 1))

    #Transform feature to Pandas DataFrame
    transform_feature = pd.DataFrame(transform_feature,
                                     columns=list(ohe_model.categories_[0]))
    transform_feature = pd.DataFrame(transform_feature)

    # Re-indexing the feature
    transform_feature.set_index(data.index, inplace=True)

    # Concatenate the transformed data to the actual dataset
    data = pd.concat([transform_feature, data], axis=1)

    # Drop the initial feature
    data.drop(columns=feature, inplace=True)
    
    return data

if __name__ == "__main__":
    # 1. Load Configuration File
    config_data = utils.load_config()

    # 2. Load Dataset
    train_set, valid_set, test_set = load_dataset(config_data=config_data)

    # 3. Removing outliers on Train Set Data
    train_set_clean = remove_outliers(set_data=train_set)

    # 4. Encode Data
    # Create list to store the results after data is transformed
    clean_dataset = []

    # Define the feature that should be transformed
    ohe_predictors = config_data["ohe_predictors"]

    # Put all sets of data (train, valid and test) into tuple
    # So, it can be iterated
    datasets = (train_set_clean, valid_set, test_set)

    # itrate through dataset and predictors
    for data in datasets:
        for i, predictor in enumerate(ohe_predictors):
            data_range = config_data[f"range_{predictor}"]

            # Fit the data with OHE
            fitted_ohe = ohe_fit(feature=predictor,
                                  data_range=data_range)
            if i == 0:
                # Transform data with OHE
                ohe_set = ohe_transform(feature=predictor,
                                        data=data,
                                        ohe_model=fitted_ohe)
            else:
                # Transform data with OHE
                ohe_set = ohe_transform(feature=predictor,
                                        data=ohe_set,
                                        ohe_model=fitted_ohe)

            # Renaming the ohe data
            ohe_set.rename(config_data[f"{predictor}_map"],
                           axis=1,
                           inplace=True)
        clean_dataset.append(ohe_set)
    
    # 5. Dropping 'atemp' Variables on all sets data
    for data in clean_dataset:
        data.drop("atemp", axis=1, inplace=True)

    # 6. Unpack all of the dataset
    train_set_clean, valid_set_clean, test_set_clean = clean_dataset

    # 7. Get the current columns
    current_columns = train_set_clean.columns.tolist()

    # 8. Dropping output columns in the current columns
    current_columns.remove(config_data["label"])

    # Dump preprocessng data to pickle
    utils.pickle_dump(train_set_clean[current_columns],
                      config_data["train_feng_set_path"][0])
    utils.pickle_dump(train_set_clean[config_data["label"]], 
                      config_data["train_feng_set_path"][1])

    # Dump Valid Set Data
    utils.pickle_dump(valid_set_clean[current_columns],
                      config_data["valid_feng_set_path"][0])
    utils.pickle_dump(valid_set_clean[config_data["label"]], 
                      config_data["valid_feng_set_path"][1])

    # Dump Test Set Data

    utils.pickle_dump(test_set_clean[current_columns],
                      config_data["test_feng_set_path"][0])
    utils.pickle_dump(test_set_clean[config_data["label"]],
                      config_data["test_feng_set_path"][1])