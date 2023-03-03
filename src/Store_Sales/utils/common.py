import os
from box.exceptions import BoxValueError
import yaml
from Store_Sales import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import pandas as pd

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns
    Args:
        path_to_yaml (str): path like input
    Raises:
        ValueError: if yaml file is empty
        e: empty file
    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data
    Args:
        path (Path): path to json file
    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories
    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def load_data(file_path: Path, schema_file_path: Path) -> pd.DataFrame:
    """_summary_

    Args:
        file_path (Path): file path of data csv file
        schema_file_path (Path): file path of schema yaml file

    Raises:
        Exception: raise error message if column is not present in data csv file
    Returns:
        pd.DataFrame:  schema columns validated pandas dataframe
    """
    
    datatset_schema = read_yaml(schema_file_path)

    schema = datatset_schema["columns"]

    dataframe = pd.read_csv(file_path)

    error_messgae = ""

    for column in dataframe.columns:
        if column!=datatset_schema['target_column']:
            if column in list(schema.keys()):
                dataframe[column]=dataframe[column].astype(schema[column])
            else:
                error_messgae = f"{error_messgae} \n Column: [{column}] is not in the schema."
    
    if len(error_messgae) > 0:
        raise Exception(error_messgae)
    return dataframe

   
    