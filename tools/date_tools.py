import functools
import calendar
from datetime import datetime, timedelta
from pandas import Timestamp
from typing import Union
from tools.general_tools import print_string_to_text_file

# List of public holidays (in yyyy-mm-dd format)
holiday_list = ['2018-01-26', '2018-02-13', '2018-03-02', '2018-03-29', '2018-03-30', '2018-05-01', '2018-08-15',
                '2018-08-22', '2018-09-13', '2018-09-20', '2018-10-02', '2018-10-18', '2018-11-07', '2018-11-08',
                '2018-11-23', '2018-12-25', '2019-03-04', '2019-03-21', '2019-04-17', '2019-04-19', '2019-04-29',
                '2019-05-01', '2019-06-05', '2019-08-12', '2019-08-15', '2019-09-02', '2019-09-10', '2019-10-02',
                '2019-10-08', '2019-10-21', '2019-10-28', '2019-11-12', '2019-12-25', '2020-02-21', '2020-03-10',
                '2020-04-02', '2020-04-06', '2020-04-10', '2020-04-14', '2020-05-01', '2020-05-25', '2020-10-02',
                '2020-11-16', '2020-11-30', '2020-12-25', '2021-01-26', '2021-03-11', '2021-03-29', '2021-04-02',
                '2021-04-14', '2021-04-21', '2021-05-13', '2021-07-21', '2021-08-19', '2021-09-10', '2021-10-15',
                '2021-11-04', '2021-11-05', '2021-11-19', '2022-01-26', '2022-03-01', '2022-03-18', '2022-04-14',
                '2022-04-15', '2022-05-03', '2022-08-09', '2022-08-15', '2022-08-31', '2022-10-05', '2022-10-24',
                '2022-10-26', '2022-11-08', '2023-01-26', '2023-03-07', '2023-03-30', '2023-04-07', '2023-04-14',
                '2023-05-01', '2023-06-29', '2023-08-15', '2023-09-19', '2023-10-02', '2023-10-24', '2023-11-14',
                '2023-11-27', '2023-12-25', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29', '2024-04-11',
                '2024-04-17', '2024-05-01', '2024-06-17', '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01',
                '2024-11-15', '2024-12-25']

# Note: '2024-11-01' has Muhurat Trading Session. The timings of this session will be notified by NSE at a later date.
# However, '2024-11-01' is included in the list 'holiday_list' for now. Check if past Diwalis from 2023 and before
# had Muhurat Trading Sessions. If they did, then remove those dates from the list 'holiday_list' above and add their
# historical data to the database. Also, add the Muhurat Trading Session timings to the database.

# There have been other random trading sessions in the past, particularly on the days of the budget as well as when
# NSE has been checking their disaster recovery systems. Handle these dates appropriately at some point.

# Note: I'm not sure if the below 2 statements are true any more.
# Update this list whenever another month passes in real time
# Do not update this in the middle of a month

holiday_list = [datetime.strptime(date_value, "%Y-%m-%d") for date_value in holiday_list]

# Creating set for faster linear searches
# noinspection PyTypeChecker
holiday_list_set = (frozenset(holiday_list)
                    | frozenset([Timestamp(f"{str(datetime_value)}+05:30") for datetime_value in holiday_list]))

# The first day of the first month for which complete month data must be stored
# This is the start day of the month after the month corresponding to the first date in the list 'holiday_list'.
# Eg. If the first date in the list 'holiday_list' is '2019-03-04', then the value of
# 'holiday_list_start_day_for_complete_data' will be '2019-04-01'.
holiday_list_start_day_for_complete_data = holiday_list[0].replace(month=holiday_list[0].month + 1, day=1)

# noinspection PyTypeChecker
holiday_list_start_day_for_complete_data_timestamp = Timestamp(f"{str(holiday_list_start_day_for_complete_data)}+05:30")

number_of_days_in_last_month = calendar.monthrange(holiday_list[-1].year, holiday_list[-1].month)[1]
# The market close time on the last day of the last month for which complete month data must be stored
holiday_list_last_day_for_complete_data_market_close = holiday_list[-1].replace(month=holiday_list[-1].month,
                                                                                day=number_of_days_in_last_month,
                                                                                hour=15, minute=30)
# noinspection PyTypeChecker
holiday_list_last_day_for_complete_data_market_close_timestamp = \
    Timestamp(f"{str(holiday_list_last_day_for_complete_data_market_close)}+05:30")

trading_dates_dict = {}

trading_days_month_dict = {}

trading_days_month_dict_list = []


def midnight(day: datetime) -> datetime:
    """
    This function takes the datetime object of a day as input and returns a datetime object representing midnight on
    that day.

    Args:
        day:
            A datetime object

    Returns:
        A datetime object representing midnight on that day.
    """
    midnight_datetime = day.replace(hour=0, minute=0, second=0, microsecond=0)

    return midnight_datetime


@functools.lru_cache(maxsize=20)
def is_trading_day(day: Union[datetime, Timestamp], holiday_list_set_local: frozenset = holiday_list_set) -> bool:
    """
    This function returns True if the given day is a trading day. Else, it returns False. It does this by checking if
    the given day is a holiday or a weekend. It determines if the day is a holiday by checking if the day lies in the
    frozenset 'holiday_list_set_local'. The value of this set is the 'holiday_list_set' global variable by default.
    This set parameter should only be provided custom values for testing purposes.

    Args:
        day:
            A datetime object that is time zone unaware or a Timestamp value that is timezone aware representing the
            day.
        holiday_list_set_local:
            A frozenset of datetime objects representing the holidays. The default value of this parameter is the
            'holiday_list_set' global variable. This parameter should only be provided custom values for testing.


    Raises:
        ValueError:
            1. If the date given as input lies outside the range of the list 'tools.date_tools.holiday_list'
    """
    is_week_day = day.isoweekday() < 6

    if is_time_zone_aware(day):
        if not (holiday_list_start_day_for_complete_data_timestamp
                <= day <= holiday_list_last_day_for_complete_data_market_close_timestamp):
            raise ValueError(f"Expand the list 'tools.date_tools.holiday_list' to cover the date '{day}' "
                             "given as input to this function")

    else:
        if not holiday_list_start_day_for_complete_data <= day <= holiday_list_last_day_for_complete_data_market_close:
            raise ValueError(f"Expand the list 'tools.date_tools.holiday_list' to cover the date '{day}' "
                             "given as input to this function")

    is_holiday = day.replace(hour=0, minute=0, second=0, microsecond=0) in holiday_list_set_local

    if is_week_day and not is_holiday:
        return True
    else:
        return False


@functools.lru_cache(maxsize=20)
def get_next_trading_day(input_day: datetime, holiday_list_set_local: frozenset = holiday_list_set) -> datetime:
    """
    This function returns the datetime object of midnight on the trading day after the day passed to it.
    It checks if a given day is a trading day internally by checking if the given day is a holiday or a weekend. It
    determines if the day is a holiday by checking if the day lies in the frozenset 'holiday_list_set_local'. The value
    of this set is the 'holiday_list_set' global variable by default. This set parameter should only be provided custom
    values for testing purposes. The 'input_day' passed to this function may or may not be a trading day.

    Args:
        input_day:
            A date time object representing the day
        holiday_list_set_local:
            A frozenset of datetime objects representing the trading holidays. The default value of this parameter is
            the 'holiday_list_set' global variable. This parameter should only be provided custom values for testing.
    """
    day = input_day.replace(hour=0, minute=0, second=0, microsecond=0)

    while True:
        day += timedelta(days=1)
        is_week_day = day.isoweekday() < 6
        is_holiday = day in holiday_list_set_local
        if is_week_day and not is_holiday:
            return day


def is_time_zone_aware(input_datetime: Union[datetime, Timestamp]) -> bool:
    """Returns True if datetime or Timestamp is timezone aware and False otherwise"""
    if len(str(input_datetime)) > 19:
        # A timezone aware value has its representation string looks like this: "2022-06-09 00:00:00+05:30"
        return True
    else:
        # A timezone unaware value has its representation string looks like this: "2022-06-09 00:00:00"
        return False


def generate_trading_days_data(start_month: str, start_year: int, end_month: str, end_year: int,
                               date_file: str = "tools/dates_list.py"):
    """
    This function defines lists of Timestamp values in the global dict 'trading_dates_dict'. Each of these lists is
    a list of the form 'dates_{month}_{year}' which contains a list of all the trading days associated with that month.
    Lists are defined for every month from the start month to the end month.

    It also defines a dictionary 'trading_days_month_dict' which contains keys of the form '{year}_{month}' and values
    that are equal to the number of trading days in that month.

    It also defines a list 'trading_days_month_dict_list' of the keys of the dict 'trading_days_month_dict'

    Here, 'trading_dates_dict', 'trading_days_month_dict' and 'trading_days_month_dict_list' are all pre-existing global
    variables whose values are set in this function.

    It also prints these global variables to the file 'date_file'. This file can be used for looking up dates during
    debugging. It is not used in this function.

    See unit tests for examples.

    Args:
        start_month:
            The start month as a 3 letter title case string. Eg. "Jan"
        start_year:
            An int representing the start year
        end_month:
            The end month as a 3 letter title case string. Eg. "Jan"
        end_year:
            An int representing the end year
        date_file:
            A string representing the name of the python file in which lists and dicts will be defined

    Raises:
        ValueError:
            This function raises a ValueError if the start date of the first month or the last date of the end month
            lie outside the date range boundary defined by the list 'holiday_list'.

    """
    global trading_dates_dict, trading_days_month_dict, trading_days_month_dict_list

    month_name_to_int = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                         "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

    start_month = month_name_to_int[start_month]
    end_month = month_name_to_int[end_month]

    first_day_of_start_month = datetime(year=start_year, month=start_month, day=1)
    last_day_of_end_month = datetime(year=end_year, month=end_month, day=calendar.monthrange(end_year, end_month)[1])

    if (first_day_of_start_month < holiday_list_start_day_for_complete_data
            or last_day_of_end_month > holiday_list_last_day_for_complete_data_market_close):
        raise ValueError

    if start_year == end_year:
        months_list_of_tuples = [(month, end_year) for month in range(start_month, end_month + 1)]

    elif end_year == (start_year + 1):
        months_list_of_tuples = ([(month, start_year) for month in range(start_month, 13)]
                                 + [(month, end_year) for month in range(1, end_month + 1)])

    else:
        months_list_of_tuples = ([(month, start_year) for month in range(start_month, 13)]
                                 + [(month, year) for year in range(start_year + 1, end_year) for month in range(1, 13)]
                                 + [(month, end_year) for month in range(1, end_month + 1)])

    # Debugging file text string
    file_string = "from pandas import Timestamp\n\n"

    for month, year in months_list_of_tuples:
        file_string += "# noinspection PyTypeChecker\n"

        first_day_of_month = Timestamp(year=year, month=month, day=1)
        if is_trading_day(first_day_of_month):
            first_trading_day_of_month = first_day_of_month
        else:
            first_trading_day_of_month = get_next_trading_day(first_day_of_month)

        month_trading_days_list = []
        current_day = first_trading_day_of_month

        outer_loop_break = False

        while True:
            month_trading_days_list.append(current_day)

            # Getting the next trading day as long as it is in the same month
            while True:
                current_day += timedelta(days=1)
                if current_day.month == (month + 1) or (current_day.month == 1 and month == 12):
                    outer_loop_break = True
                    break
                if is_trading_day(current_day):
                    break

            if outer_loop_break:
                break

        date_strings_list = [f"Timestamp('{trading_day :%Y-%m-%d %H:%M:%S}+0530')" for trading_day in
                             month_trading_days_list]

        file_string += f"dates_{month}_{year} = [{', '.join(date_strings_list)}]\n\n"

        # Converting datetimes without time zones to Timestamps with time zones
        # noinspection PyTypeChecker
        date_timestamps_list = [Timestamp(f'{trading_day :%Y-%m-%d %H:%M:%S}+0530') for trading_day in
                                month_trading_days_list]

        trading_dates_dict[f"dates_{month}_{year}"] = date_timestamps_list

        trading_days_month_dict[f"{year}_{month}"] = len(month_trading_days_list)

    trading_days_month_dict_list = list(trading_days_month_dict)

    file_string += "trading_days_month_dict = " + str(trading_days_month_dict) + "\n\n"

    file_string += "trading_days_month_dict_list = list(trading_days_month_dict)"

    print_string_to_text_file(file_string, date_file)


month_number_to_name = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

# By running this function every time date_tools is imported, we make sure that the only thing that needs to be updated
# from time to time is holiday_list which needs to be updated every month. Whenever holiday_list is updated, everything
# else in the code will change automatically.

generate_trading_days_data(month_number_to_name[holiday_list_start_day_for_complete_data_timestamp.month],
                           holiday_list_start_day_for_complete_data_timestamp.year,
                           month_number_to_name[holiday_list_last_day_for_complete_data_market_close_timestamp.month],
                           holiday_list_last_day_for_complete_data_market_close_timestamp.year)

@functools.lru_cache(maxsize=20)
def get_previous_trading_day(trading_day: datetime, holiday_list_set_local: frozenset = holiday_list_set) -> datetime:
    """
    This function returns the datetime object of midnight on the trading day previous to the day passed to it. The day
    passed as input may or may not be a trading day.

    It checks if a given day is a trading day internally by checking if the given day is a holiday or a weekend. It
    determines if the day is a holiday by checking if the day lies in the frozenset 'holiday_list_set_local'. The value
    of this set is the 'holiday_list_set' global variable by default. This set parameter should only be provided custom
    values for testing purposes.

    Args:
        trading_day:
            A date time object representing the day
        holiday_list_set_local:
            A frozenset of datetime objects representing the trading holidays. The default value of this parameter is
            the 'holiday_list_set' global variable. This parameter should only be provided custom values for testing.

    Returns:
        The datetime object representing midnight on the previous trading day
    """
    trading_day = trading_day.replace(hour=0, minute=0, second=0, microsecond=0)
    while True:
        trading_day = trading_day - timedelta(days=1)
        is_week_day = trading_day.isoweekday() < 6
        is_holiday = trading_day in holiday_list_set_local
        if is_week_day and not is_holiday:
            return trading_day


def get_list_of_trading_days(start_day_datetime: datetime, end_day_datetime: datetime) -> list[
    datetime]:
    """
    This function returns the list of trading days from midnight of 'start_day' to midnight of 'end_day', both
    inclusive.

    Args:
        start_day_datetime:
            A datetime value representing the start day
        end_day_datetime:
            A datetime value representing the end day

    Returns:
        If 'start_day' comes on or before 'end_day', this function returns a list
        of trading days from 'start_day' to 'end_day', both inclusive .

    Raises:
        ValueError:
            If 'start_day' is after 'end_day', this function will raise a ValueError
    """
    start_day_datetime = midnight(start_day_datetime)
    end_day_datetime = midnight(end_day_datetime)

    if start_day_datetime > end_day_datetime:
        raise ValueError(f"Start date '{start_day_datetime}' is after end date '{end_day_datetime}'")

    if is_trading_day(end_day_datetime):
        last_trading_day = end_day_datetime
    else:
        last_trading_day = get_previous_trading_day(end_day_datetime)

    trading_day = last_trading_day

    trading_days_list = []

    while trading_day >= start_day_datetime:
        is_week_day = trading_day.isoweekday() < 6
        is_holiday = trading_day in holiday_list_set
        if is_week_day and not is_holiday:
            trading_days_list.append(trading_day)
        trading_day = trading_day - timedelta(days=1)

    trading_days_list_in_ascending_order = trading_days_list[::-1]

    return trading_days_list_in_ascending_order