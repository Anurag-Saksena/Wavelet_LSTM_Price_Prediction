import numpy as np
from pandas import DataFrame
from scipy.sparse import linalg
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

numeric_indicator_columns = ["sma", "ema", "rsi", "atrts", "ehler_fisher", "true_range", "average_true_range",
                             "supertrend", "bbands_upper", "bbands_lower"]

boolean_indicator_columns = ["ehler_fisher_greater_than_0", "ehler_fisher_less_than_0", "rsi_greater_than_70",
                             "rsi_less_than_30", "close_crosses_above_supertrend", "close_crosses_below_supertrend",
                             "close_crosses_above_atrts", "close_crosses_below_atrts",
                             "close_crosses_above_bbands_upper", "close_crosses_below_bbands_lower",
                             "close_crosses_above_sma", "close_crosses_below_sma", "close_crosses_above_ema",
                             "close_crosses_below_ema", "tr_less_than_atr"]


def generate_training_and_testing_data(training_data: DataFrame, testing_data: DataFrame, lookback_period: int):
    training_data = training_data.values
    testing_data = testing_data.values

    scaler = MinMaxScaler(feature_range=(0, 1))

    # print(training_data.shape)
    # print(testing_data.shape)

    # transform train
    training_data = scaler.fit_transform(training_data.reshape(-1, 1))

    # transform test
    testing_data = scaler.transform(testing_data.reshape(-1, 1))

    # print(training_data.shape)
    # print(testing_data.shape)

    # Creating the training and testing data sets
    x_train = []
    y_train = []

    for i in range(lookback_period, len(training_data)):
        x_train.append(training_data[i - lookback_period:i, 0])
        y_train.append(training_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # print(x_train.shape)
    # print(y_train.shape)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # print(x_train.shape)

    x_test = []
    y_test = []

    for i in range(lookback_period, len(testing_data)):
        x_test.append(testing_data[i - lookback_period:i, 0])
        y_test.append(testing_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return scaler, (x_train, y_train), (x_test, y_test)


def generate_training_and_testing_data_standard_scaler(training_data: DataFrame, testing_data: DataFrame,
                                                       lookback_period: int):
    training_data = training_data.values
    testing_data = testing_data.values

    scaler = StandardScaler()

    # print(training_data.shape)
    # print(testing_data.shape)

    # transform train
    training_data = scaler.fit_transform(training_data.reshape(-1, 1))

    # transform test
    testing_data = scaler.transform(testing_data.reshape(-1, 1))

    # print(training_data.shape)
    # print(testing_data.shape)

    # Creating the training and testing data sets
    x_train = []
    y_train = []

    for i in range(lookback_period, len(training_data)):
        x_train.append(training_data[i - lookback_period:i, 0])
        y_train.append(training_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # print(x_train.shape)
    # print(y_train.shape)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # print(x_train.shape)

    x_test = []
    y_test = []

    for i in range(lookback_period, len(testing_data)):
        x_test.append(testing_data[i - lookback_period:i, 0])
        y_test.append(testing_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return scaler, (x_train, y_train), (x_test, y_test)


def featureSelect_dataframe(X, y, criteria, k):
    # initialize our function/method
    reg = SelectKBest(criteria, k=k).fit(X, y)

    # transform after creating the reg (so we can use getsupport)
    reg.transform(X)

    # filter down X based on kept columns
    X = X[[val for i, val in enumerate(X.columns) if reg.get_support()[i]]]

    # return that dataframe
    return X


def generate_training_and_testing_data_multiple_features(training_data_old: DataFrame, testing_data_old: DataFrame,
                                                         lookback_period: int, number_of_features: int):
    # print("Output 1")
    # print(training_data_old.shape)
    # print(testing_data_old.shape)
    #
    # print(training_data_old.shape[1])

    training_data_old = training_data_old.copy()
    testing_data_old = testing_data_old.copy()

    for column in list(training_data_old.columns):
        if set(training_data_old[column]) == {0, 1}:
            d = {0: 0.2, 1: 0.8}
            training_data_old[column] = training_data_old[column].apply(lambda x: d[x])

    for column in list(testing_data_old.columns):
        if set(testing_data_old[column]) == {0, 1}:
            d = {0: 0.2, 1: 0.8}
            testing_data_old[column] = testing_data_old[column].apply(lambda x: d[x])

    for column in list(training_data_old.columns):
        scaler = MinMaxScaler(feature_range=(0, 1))
        # transform train
        training_data_old.loc[:, column] = scaler.fit_transform(training_data_old.loc[:, column].values.reshape(-1, 1))

        # transform test
        testing_data_old.loc[:, column] = scaler.transform(testing_data_old.loc[:, column].values.reshape(-1, 1))

    training_data = featureSelect_dataframe(training_data_old, training_data_old[["close"]],
                                            f_regression, number_of_features)

    # training_data = training_data_old[random.sample(list(training_data_old.columns), number_of_features)]

    training_data.loc[:, 'close'] = training_data_old['close']

    print("Selected Features: ", list(training_data.columns))
    testing_data = testing_data_old[list(training_data.columns)]

    training_data = training_data.values
    testing_data = testing_data.values

    # print("Output 2")
    #
    # print(training_data.shape)
    # print(testing_data.shape)

    # print("Output 3")

    # print(training_data.shape)
    # print(testing_data.shape)
    #
    # # transform train
    # training_data = scaler.fit_transform(training_data)
    #
    # # transform test
    # testing_data = scaler.transform(testing_data)

    # print("Output 4")
    #
    # print(training_data.shape)
    # print(testing_data.shape)

    # Creating the training and testing data sets
    x_train = []
    y_train = []

    for i in range(lookback_period, len(training_data)):
        x_train.append(training_data[i - lookback_period:i, :])
        y_train.append(training_data[i, -1])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # print("Output 5")
    #
    # print(x_train.shape)
    # print(y_train.shape)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    # print("Output 6")
    # print(x_train.shape)

    x_test = []
    y_test = []

    for i in range(lookback_period, len(testing_data)):
        x_test.append(testing_data[i - lookback_period:i, :])
        y_test.append(testing_data[i, -1])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # print("Output 7")
    #
    # print(x_test.shape)
    # print(y_test.shape)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # print("Output 8")
    #
    # print(x_test.shape)

    return scaler, (x_train, y_train), (x_test, y_test)


def select_data_columns(df: DataFrame, just_close: bool = False, just_open_and_close: bool = False,
                        just_ohlc: bool = False, just_ohlcvoi: bool = False, numerical_indicators_present: bool = False,
                        boolean_indicators_present: bool = False) -> DataFrame:
    price_column_parameters = [just_close, just_open_and_close, just_ohlc, just_ohlcvoi]
    price_column_count = sum(price_column_parameters)

    if price_column_count != 1:
        raise ValueError("Exactly one of the parameters 'just_close', 'just_open_and_close', 'just_ohlc', and "
                         "'just_ohlcvoi' should be set to 'True'")

    price_columns = []

    if just_close:
        price_columns = ["close"]
    elif just_open_and_close:
        price_columns = ["open", "close"]
    elif just_ohlc:
        price_columns = ["open", "high", "low", "close"]
    elif just_ohlcvoi:
        price_columns = ["open", "high", "low", "close", "volume", "oi"]

    columns = price_columns

    if numerical_indicators_present:
        columns += numeric_indicator_columns

    if boolean_indicators_present:
        columns += boolean_indicator_columns

    return df[columns]
