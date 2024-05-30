import os

import pandas as pd
from pandas import DataFrame

from project_config import project_path


def save_df_to_csv(input_df: DataFrame, csv_file_path: str) -> None:
    """
    This function loads the contents of the input DataFrame into a csv file stored at path 'csv_file_path'. If a csv
    file already exists at path 'csv_file_path', then it is deleted and replaced.

    Args:
        csv_file_path:
            A string representing path of csv file with respect to project path
        input_df:
            The input DataFrame to be stored as a csv file

    Returns:
        None
    """
    current_path = os.getcwd()

    os.chdir(project_path)

    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)

    input_df.to_csv(csv_file_path, index=False)

    os.chdir(current_path)


def load_time_series_csv_to_df(csv_file_path: str, date_column_name: str = "date",
                               column_data_types: dict = None) -> DataFrame:
    """
    This function loads the contents of the time series csv file at path 'csv_file_path' into a pandas Dataframe and
    returns it. The function reads the values in the date column of the csv file and converts them to Timestamp values
    in the DataFrame.

    Args:
        date_column_name:
            Name of column in csv that contains date or time data
        csv_file_path:
            A string representing path of csv file with respect to project path
        column_data_types:
            A dict representing the data types of the column in terms of numpy data types
            Eg.{"oi": np.int32, "volume": np.int32, "open": np.float32, "close": np.float32,
                "low": np.float32, "high": np.float32}
    """

    current_path = os.getcwd()

    os.chdir(project_path)

    if column_data_types is not None:
        df = pd.read_csv(csv_file_path, parse_dates=[date_column_name], engine="c", dtype=column_data_types)
    else:
        df = pd.read_csv(csv_file_path, parse_dates=[date_column_name], engine="c")

    os.chdir(current_path)

    return df

