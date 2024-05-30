"""
This module contains various utility functions for dealing with lists
and DataFrames
"""
import os
from typing import Union

import pandas as pd
from pandas import Timestamp, DataFrame
from datetime import datetime

from project_config import project_path


def print_string_to_text_file(input_string: str, text_file_path: str) -> None:
    """
    This function prints the string 'input_string' to the text file with path
    'text_file_path' where the path 'text_file_path' is defined with respect to
    project path. If a text file already exists at this path, it is deleted.
    """
    current_path = os.getcwd()

    os.chdir(project_path)

    if os.path.exists(text_file_path):
        os.remove(text_file_path)

    with open(text_file_path, 'w') as txt:
        txt.write(input_string)

    os.chdir(current_path)


def print_string_to_text_file_append(input_string: str, text_file_path: str) -> None:
    """
    This function prints the string 'input_string' to the text file with path 'text_file_path' where the path
    'text_file_path' is defined with respect to project path. If a text file already exists at this path, the text of
    'input_string' is appended to the content of the pre-existing text file.
    """
    current_path = os.getcwd()

    os.chdir(project_path)

    with open(text_file_path, 'a') as txt:
        txt.write(input_string)

    os.chdir(current_path)


def time_stamp_to_string(time_stamp: Union[Timestamp, datetime], time_data: bool = False,
                         second_data: bool = False) -> str:
    """
    This function converts a TimeStamp or datetime object into a date string.
    Eg. Timestamp("2021-07-28 00:00:00") -> 'Jul 28, 2021'

    Args:
        time_stamp:
            The Timestamp or datetime object to be converted to a string
        time_data:
            A boolean. If 'time_data' is set to True, the date string
            generated will also contain hours and minutes
        second_data:
            A boolean. If 'second_data' is set to True, the date string
            generated will contain hours, minutes and seconds

    Returns:
        A date string representation of the input Timestamp/datetime object.
        If 'time_data' and 'second_data' are False, the date string generated
        looks like this: 'Jul 28, 2021'. If 'time_data' is True and 'second_data'
        is False, the date string generated looks like this: 'Jul 28, 2021 12:30'.
        If 'second_data' is True, then the date string generated looks like this:
        'Jul 28, 2021 12:30:17'.

    """
    year = time_stamp.year
    month = time_stamp.month_name()[:3]
    day = time_stamp.day
    if day < 10:
        day = f"0{day}"

    time_stamp_string = f'{month} {day}, {year}'

    # There is no point in displaying seconds if hours and minutes
    # are not visible, so in case, 'second_data' is True, 'time_data'
    # will be set to True as well.

    if second_data:
        time_data = True

    if time_data:
        time_string = str(time_stamp)[11:16]
        time_stamp_string += " " + time_string
        if second_data:
            time_stamp_string += str(time_stamp)[16:19]

    return time_stamp_string


def format_dict(input_dict, suppress_lower: bool, suppress_upper: bool) -> str:
    """
    Returns a string with the items of the input dictionary as f'{key} = {value}' strings line by line. If the input
    dict keys are Timestamps or datetime objects, then they are converted to date strings in the output string.

    Args:
        input_dict:
            The dictionary to be clean printed
        suppress_lower:
            If the parameter 'suppress_lower' is set to 'True', the line made up of 100 '-' characters
            at the bottom of the formatted dict output will not be printed.
        suppress_upper:
            If the parameter 'suppress_upper' is set to 'True', the line made up of 100 '-' characters
            at the top of the formatted dict output will not be printed.
    """
    if len(input_dict) == 0:
        return ''

    clean_string = ''

    if not suppress_upper:
        clean_string += '-' * 100 + '\n'

    type_of_dict_keys = type(list(input_dict.keys())[0])

    # Converting dictionary keys that are Timestamp or datetime objects to date strings before printing

    if type_of_dict_keys == Timestamp or type_of_dict_keys == datetime:
        input_dict = {time_stamp_to_string(key, time_data=True, second_data=True): value
                      for key, value in input_dict.items()}

    clean_string += '\n\n'.join([f'{key}   =   {value}' for key, value in input_dict.items()]) + '\n'

    if not suppress_lower:
        clean_string += '-' * 100 + '\n'

    return clean_string


def format_list_or_tuple_or_set(input_iterable, suppress_lower: bool, suppress_upper: bool) -> str:
    """
    Returns a string with the items of the input list or tuple or set present line by line

    Args:
        input_iterable:
            The list or tuple or set to be clean printed
        suppress_lower:
            If the parameter 'suppress_lower' is set to 'True', the line made up of 100 '-' characters
            at the bottom of the formatted output will not be printed.
        suppress_upper:
            If the parameter 'suppress_upper' is set to 'True', the line made up of 100 '-' characters
            at the top of the formatted output will not be printed.
    """
    if len(input_iterable) == 0:
        return ''

    clean_string = ''

    if not suppress_upper:
        clean_string += '-' * 100 + '\n'

    clean_string += '\n\n'.join([str(item) for item in input_iterable]) + '\n'

    if not suppress_lower:
        clean_string += '-' * 100 + '\n'

    return clean_string


def format_output(func_input, suppress_lower: bool, suppress_upper: bool) -> str:
    """
    Returns a formatted output string:

    1. If the input is a dictionary, the formatted output string will be as f'{key} = {value}' strings line by line
    as returned by format_dict().

    2. If the input is a list or tuple or set, the formatted output string will consist of the iterable's
    items getting printed as strings line by line

    Args:
        func_input:
            the input list or set or tuple or dict
        suppress_lower:
            If the parameter 'suppress_lower' is set to 'True', the line made up of 100 '-' characters
            at the bottom of the formatted list/tuple/dict/set output will not be printed.
        suppress_upper:
            If the parameter 'suppress_upper' is set to 'True', the line made up of 100 '-' characters
            at the top of the formatted list/tuple/dict/set output will not be printed.

    Returns:
        A formatted output string for the input iterable

    Raises:
        TypeError:
            This function raises a TypeError if the input 'func_input' isn't a list, tuple, set or dict

    """

    if type(func_input) == dict:
        return format_dict(func_input,
                           suppress_lower=suppress_lower,
                           suppress_upper=suppress_upper)

    elif type(func_input) in [list, tuple, set]:
        return format_list_or_tuple_or_set(func_input,
                                           suppress_lower=suppress_lower,
                                           suppress_upper=suppress_upper)

    else:
        raise TypeError("The input 'func_input' must be of type list, tuple, set or dict")


def format_print(func_input, suppress_lower: bool = False, suppress_upper: bool = False,
                 file_path: str = None) -> None:
    """
    Prints a formatted output string:

    1. If the input is a dictionary, the formatted output string will be as f'{key} = {value}' strings line by line
    (in line with the string returned by format_dict())

    2. If the input is a list or tuple or set, the formatted output string will consist of the iterable's
    items getting printed as strings line by line

    Args:
        func_input:
            the input list or set or tuple or dict
        suppress_lower:
            If the parameter 'suppress_lower' is set to 'True', the line made up of 100 '-' characters
            at the bottom of the formatted list/tuple/dict/set output will not be printed.
        suppress_upper:
            If the parameter 'suppress_upper' is set to 'True', the line made up of 100 '-' characters
            at the top of the formatted list/tuple/dict/set output will not be printed.
        file_path:
            If 'file_path' is not None, then it is a string representing the path of the text file to which
            the formatted output string is to be printed. This path is defined with respect to project path.

    Returns:
        None

    """
    formatted_string = format_output(func_input=func_input,
                                     suppress_lower=suppress_lower,
                                     suppress_upper=suppress_upper)

    if file_path is not None:
        print_string_to_text_file(formatted_string, file_path)
    else:
        print(formatted_string)


def format_print_append(func_input, file_path: str, suppress_lower: bool = False, suppress_upper: bool = False) -> None:
    """
    This function has the same exact functionality as format_print() but it appends 2 new line characters followed by
    the output to the text file provided as input.
    """
    formatted_string = format_output(func_input=func_input,
                                     suppress_lower=suppress_lower,
                                     suppress_upper=suppress_upper)

    print_string_to_text_file_append("\n\n" + formatted_string, file_path)


transaction_log = format_print_append


def print_full(df: DataFrame, num: int = 2, truncated: bool = True, true_date: bool = False, rounding: bool = True,
               date_column_name: str = "date", time_data: bool = True, file_path: str = None,
               append_mode: bool = False) -> None:
    """
    Pretty prints pandas dataframe with all columns displayed

    This function does not modify the input DataFrame.

    The parameter 'true_date' when set to 'True' will cause the function to display the dates in the DataFrame as
    Timestamp objects. Otherwise, they will be displayed as date strings in accordance with the time_stamp_to_string()
    function with the 'time_data' parameter passed to the time_stamp_to_string() function equal to the 'time_data'
    parameter passed to this function.

    Args:
        rounding:
            If set to True, all float values in the DataFrame will be rounded
        df:
            The DataFrame to be pretty printed. Note that this function creates a
            deep copy of this DataFrame, modifies it and then prints it. It does not
            change the original DataFrame.
        num:
            The parameter 'num' is an integer with default value '2'. It sets the number
            of decimal places to which float values in the dataframe will be rounded.
        truncated:
            A boolean that is set to True by default. This parameter when set to True,
            will cause this function to print a truncated version of this pandas DataFrame
            with only the first and last 5 rows visible if number of rows in DataFrame is large.
            If number of rows in DataFrame is small, then all rows of the DataFrame will be displayed
            even if truncated is set to True. This parameter when set to False,
            will cause this function to print the entire DataFrame with all rows visible.
        true_date:
            A boolean that is set to False by default. Set this parameter to True if no column in your DataFrame has
            date values. Now, if column 'date_column_name' in 'df' has date values, then this parameter when set to
            True, will cause this function to print the actual Timestamp objects in the DataFrame. This parameter, when
            set to False, will cause this function to convert all Timestamp objects in the DataFrame to date strings in
            accordance with the time_stamp_to_string() function.
        date_column_name:
            A string that represents the name of the column containing datetime or timestamp
            values that this function converts to date strings if the 'true_date' parameter is set to False
        time_data:
            A boolean that is set to False by default. This parameter when set to True,
            will cause this function to print hours, minutes and seconds in the corresponding converted date
            strings as well.
        file_path:
            If 'file_path' is not None, then it is a string representing the path of the text file to which
            the formatted output string is to be printed. This path is defined with respect to project path.
        append_mode:
            If 'file_path' is not None and 'append_mode' is set to False, then the formatted output is printed to a new
            text file whose path is given by 'file_path'. If a file already exists at the same path, it is deleted. On
            the other hand, if 'file_path' is not None and 'append_mode' is set to True, then the formatted output is
            appended to the content of the text file present at path 'file_path'. If no text file is present at path
            'file_path', then a new one is created at this path with the formatted output.

    Returns:
        None

    """
    if num < 0:
        raise ValueError(f"The input 'num' must be >=0")

    if not truncated:
        pd.set_option('display.max_rows', None)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)

    if rounding:
        pd.set_option('display.float_format', f'{{:20,.{num}f}}'.format)

    pd.set_option('display.max_colwidth', None)

    # Creating a deep copy of the dataframe so that when we convert
    # the Timestamps in the 'date' column to date strings, these
    # changes aren't reflected in the original DataFrame

    df_copy = df.copy(deep=True)

    # Converting Timestamps in the 'date' column to date strings

    # Assigning second_data=time_data because if the 'time_data' parameter of print_full() is set to True,
    # then second data should also be displayed

    if not true_date:
        df_copy[date_column_name] = [
            time_stamp_to_string(time_stamp=time_stamp, time_data=time_data, second_data=time_data)
            for time_stamp in list(df_copy[date_column_name])]

    if file_path is not None:
        if not append_mode:
            print_string_to_text_file(str(df_copy), file_path)
        else:
            print_string_to_text_file_append(str(df_copy), file_path)
    else:
        print(df_copy)

    if not truncated:
        pd.reset_option('display.max_rows')

    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def read_string_from_text_file(text_file_path: str) -> str:
    """
    This function reads the content of the text file stored at path 'text_file_path' into a string
    and returns it. The path 'text_file_path' here is defined with respect to project path.
    """
    current_path = os.getcwd()

    os.chdir(project_path)

    with open(text_file_path) as txt:
        output_string = txt.read()

    os.chdir(current_path)

    return output_string
