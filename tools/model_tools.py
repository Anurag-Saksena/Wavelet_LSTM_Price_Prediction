import itertools
import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, layers
from xgboost import XGBRegressor

from tools.data_tools import generate_training_and_testing_data, generate_training_and_testing_data_multiple_features
from tools.metrics_tools import calculate_prediction_percentage_within_threshold


def construct_lstm_price_prediction_model(batch_size: int, num_epochs: int, x_train: np.ndarray,
                                          y_train: np.ndarray, dense_units: int, lstm_units: int,
                                          x_test: np.ndarray, y_test: np.ndarray) -> Sequential:
    """
    This function constructs an LSTM model for predicting stock prices.

    Args:
        lookback_period:
            The number of time steps to look back
        batch_size:
            The number of samples per gradient update
        num_epochs:
            The number of epochs to train the model
        x_train:
            The training input data
        y_train:
            The training output data
        x_test:
            The testing input data
        y_test:
            The testing output data

    Returns:
        The LSTM model
    """
    from keras.src.callbacks import EarlyStopping

    early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    model = Sequential()

    model.add(layers.LSTM(lstm_units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(lstm_units, return_sequences=False))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(dense_units))
    model.add(layers.Dense(dense_units // 2))

    model.add(layers.Dense(1))

    print(model.summary())

    model.compile(optimizer='adam', loss='mean_squared_error')

    history=model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, callbacks=[early_stopping],
              validation_data=[x_test, y_test])

    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    return model


def predict_lstm_prices(model: Sequential, scaler: MinMaxScaler, x_test: np.ndarray) -> np.ndarray:
    """
    This function predicts the stock prices using the LSTM model.

    Args:
        model:
            The LSTM model
        x_test:
            The testing input data

    Returns:
        The predicted stock prices
    """
    predicted_prices = model.predict(x_test)

    predicted_prices = scaler.inverse_transform(predicted_prices)

    return predicted_prices


def test_lstm_model(batch_size: int, num_epochs: int, x_train_list: list, dense_units: int,
                    lstm_units: int, x_test_list: list, lookback_period: int) -> dict:
    data = {}

    dataset_number = 1

    for x_train_original, x_test_original in zip(x_train_list, x_test_list):
        scaler, (x_train, y_train), (x_test, y_test) = (
            generate_training_and_testing_data(training_data=x_train_original, testing_data=x_test_original,
                                               lookback_period=lookback_period))

        rmse_list = []
        percentage_01_accuracy_list = []
        percentage_1_accuracy_list = []

        for i in range(3):
            print(f"Dataset {dataset_number} - Testing_Iteration {i + 1}")

            model = construct_lstm_price_prediction_model(batch_size=batch_size, num_epochs=num_epochs, x_train=x_train,
                                                          y_train=y_train, dense_units=dense_units,
                                                          lstm_units=lstm_units)

            print("Model trained")

            predicted_prices = predict_lstm_prices(model=model, scaler=scaler, x_test=x_test)

            predicted_prices = np.squeeze(predicted_prices)

            y_test = x_test_original[lookback_period:]

            rmse = np.sqrt(np.mean(np.power((predicted_prices - y_test), 2)))

            predicted_prices = list(predicted_prices)

            percentage_01_accuracy = calculate_prediction_percentage_within_threshold(predicted_values=predicted_prices,
                                                                                      actual_values=y_test,
                                                                                      threshold_percentage=0.1)

            percentage_1_accuracy = calculate_prediction_percentage_within_threshold(predicted_values=predicted_prices,
                                                                                     actual_values=y_test,
                                                                                     threshold_percentage=1)

            rmse_list.append(rmse)
            percentage_01_accuracy_list.append(percentage_01_accuracy)
            percentage_1_accuracy_list.append(percentage_1_accuracy)

        rmse = np.mean(rmse_list)
        percentage_01_accuracy = np.mean(percentage_01_accuracy_list)
        percentage_1_accuracy = np.mean(percentage_1_accuracy_list)

        data[f"dataset_{dataset_number}"] = {"rmse": round(rmse, 2),
                                             "percentage_01_accuracy": round(percentage_01_accuracy, 2),
                                             "percentage_1_accuracy": round(percentage_1_accuracy, 2)}

        dataset_number += 1

    return data


def optimize_lstm_model_random_search(batch_size_list: list[int], num_epochs: int, dense_units_list: list[int],
                                      lstm_units_list: list[int], x_train_list: list, x_test_list: list,
                                      lookback_period_list: list[int],
                                      number_of_iterations: int) -> dict:
    data = {}

    cross_product = list(itertools.product(batch_size_list, dense_units_list, lstm_units_list, lookback_period_list))

    selected_hyperparameters = random.sample(cross_product, number_of_iterations)

    iteration = 1

    for batch_size, dense_units, lstm_units, lookback_period in selected_hyperparameters:
        print(f"Random Search Iteration {iteration}")
        print(f"Selected Hyperparameters: batch_size = {batch_size}, dense_units = {dense_units}, "
              f"lstm_units = {lstm_units}, lookback_period = {lookback_period}")

        data[("batch_size", batch_size), ("dense_units", dense_units), ("lstm_units", lstm_units),
        ("lookback_period", lookback_period)] = test_lstm_model(batch_size=batch_size, num_epochs=num_epochs,
                                                                x_train_list=x_train_list, dense_units=dense_units,
                                                                lstm_units=lstm_units,
                                                                x_test_list=x_test_list,
                                                                lookback_period=lookback_period)

        print(data[("batch_size", batch_size), ("dense_units", dense_units), ("lstm_units", lstm_units),
        ("lookback_period", lookback_period)])

        iteration += 1

    return data


def construct_lstm_price_prediction_model_multiple_features(batch_size: int, num_epochs: int, x_train: np.ndarray,
                                                            y_train: np.ndarray, dense_units: int,
                                                            lstm_units: int) -> Sequential:
    """
    This function constructs an LSTM model for predicting stock prices.

    Args:
        lookback_period:
            The number of time steps to look back
        batch_size:
            The number of samples per gradient update
        num_epochs:
            The number of epochs to train the model
        x_train:
            The training input data
        y_train:
            The training output data
        x_test:
            The testing input data
        y_test:
            The testing output data

    Returns:
        The LSTM model
    """
    model = Sequential()

    model.add(layers.LSTM(lstm_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.Dropout(0.2))

    # model.add(layers.LSTM(lstm_units, return_sequences=False))
    model.add(layers.LSTM(lstm_units // 2, return_sequences=False))
    model.add(layers.LeakyReLU(alpha=0.3))
    # model.add(layers.Dropout(0.2))

    model.add(layers.Dense(dense_units))
    model.add(layers.Dense(dense_units // 2))

    model.add(layers.Dense(1))

    print(model.summary())

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)

    return model


def predict_lstm_prices_multiple_features(model: Sequential, scaler: MinMaxScaler, x_test: np.ndarray) -> np.ndarray:
    """
    This function predicts the stock prices using the LSTM model.

    Args:
        model:
            The LSTM model
        x_test:
            The testing input data

    Returns:
        The predicted stock prices
    """
    predicted_prices = model.predict(x_test)

    predicted_prices_full = np.zeros((predicted_prices.shape[0], x_test.shape[2]))

    predicted_prices_full[:, -1] = predicted_prices[:, 0]

    predicted_prices = scaler.inverse_transform(predicted_prices_full)[:, -1]

    return predicted_prices


def test_lstm_model_multiple_features(batch_size: int, num_epochs: int, x_train_list: list, dense_units: int,
                                      lstm_units: int, x_test_list: list, lookback_period: int, number_of_features: int
                                      ) -> dict:
    data = {}

    dataset_number = 1

    for x_train_original, x_test_original in zip(x_train_list, x_test_list):
        scaler, (x_train, y_train), (x_test, y_test) = (
            generate_training_and_testing_data_multiple_features(training_data_old=x_train_original,
                                                                 testing_data_old=x_test_original,
                                                                 lookback_period=lookback_period,
                                                                 number_of_features=number_of_features))

        rmse_list = []
        percentage_01_accuracy_list = []
        percentage_1_accuracy_list = []

        for i in range(3):
            print(f"Dataset {dataset_number} - Testing_Iteration {i + 1}")

            model = construct_lstm_price_prediction_model_multiple_features(batch_size=batch_size,
                                                                            num_epochs=num_epochs, x_train=x_train,
                                                                            y_train=y_train, dense_units=dense_units,
                                                                            lstm_units=lstm_units)

            print("Model trained")

            predicted_prices = predict_lstm_prices_multiple_features(model=model, scaler=scaler, x_test=x_test)

            predicted_prices = np.squeeze(predicted_prices)

            y_test = x_test_original[lookback_period:]["close"]

            rmse = np.sqrt(np.mean(np.power((predicted_prices - y_test), 2)))

            predicted_prices = list(predicted_prices)

            percentage_01_accuracy = calculate_prediction_percentage_within_threshold(predicted_values=predicted_prices,
                                                                                      actual_values=y_test,
                                                                                      threshold_percentage=0.1)

            percentage_1_accuracy = calculate_prediction_percentage_within_threshold(predicted_values=predicted_prices,
                                                                                     actual_values=y_test,
                                                                                     threshold_percentage=1)

            rmse_list.append(rmse)
            percentage_01_accuracy_list.append(percentage_01_accuracy)
            percentage_1_accuracy_list.append(percentage_1_accuracy)

        rmse = np.mean(rmse_list)
        percentage_01_accuracy = np.mean(percentage_01_accuracy_list)
        percentage_1_accuracy = np.mean(percentage_1_accuracy_list)

        data[f"dataset_{dataset_number}"] = {"rmse": round(rmse, 2),
                                             "percentage_01_accuracy": round(percentage_01_accuracy, 2),
                                             "percentage_1_accuracy": round(percentage_1_accuracy, 2)}

        dataset_number += 1

    return data


def optimize_lstm_model_random_search_multiple_features(batch_size_list: list[int], num_epochs: int,
                                                        dense_units_list: list[int],
                                                        lstm_units_list: list[int], x_train_list: list,
                                                        x_test_list: list,
                                                        lookback_period_list: list[int],
                                                        number_of_iterations: int,
                                                        number_of_features_list: list[int]
                                                        ) -> dict:
    data = {}

    cross_product = list(itertools.product(batch_size_list, dense_units_list, lstm_units_list, lookback_period_list,
                                           number_of_features_list))

    selected_hyperparameters = random.sample(cross_product, number_of_iterations)

    iteration = 1

    for batch_size, dense_units, lstm_units, lookback_period, number_of_features in selected_hyperparameters:
        print(f"Random Search Iteration {iteration}")
        print(f"Selected Hyperparameters: batch_size = {batch_size}, dense_units = {dense_units}, "
              f"lstm_units = {lstm_units}, lookback_period = {lookback_period}, number_of_features = {number_of_features}")

        data[("batch_size", batch_size), ("dense_units", dense_units), ("lstm_units", lstm_units),
        ("lookback_period", lookback_period), ("number_of_features", number_of_features)
        ] = test_lstm_model_multiple_features(batch_size=batch_size,
                                              num_epochs=num_epochs,
                                              x_train_list=x_train_list,
                                              dense_units=dense_units,
                                              lstm_units=lstm_units,
                                              x_test_list=x_test_list,
                                              lookback_period=lookback_period,
                                              number_of_features=number_of_features)

        print(data[("batch_size", batch_size), ("dense_units", dense_units), ("lstm_units", lstm_units),
        ("lookback_period", lookback_period), ("number_of_features", number_of_features)])

        iteration += 1

    return data


def construct_xgboost_price_prediction_model(batch_size: int, num_epochs: int, x_train: np.ndarray,
                                             y_train: np.ndarray, n_estimators: int, max_depth: int,
                                             grow_policy: str, colsample_bytree: float,
                                             reg_lambda: float) -> XGBRegressor:
    """
    This function constructs an LSTM model for predicting stock prices.

    Args:
        lookback_period:
            The number of time steps to look back
        batch_size:
            The number of samples per gradient update
        num_epochs:
            The number of epochs to train the model
        x_train:
            The training input data
        y_train:
            The training output data
        x_test:
            The testing input data
        y_test:
            The testing output data

    Returns:
        The LSTM model
    """
    model = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, max_depth=max_depth,
                         grow_policy=grow_policy, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
                         random_state=random.randint(0, 1000))

    # model = XGBRegressor(n_estimators=10,max_depth=100)

    x_train = np.squeeze(x_train)

    model.fit(x_train, y_train)

    return model


def predict_xgboost_prices(model: Sequential, scaler: MinMaxScaler, x_test: np.ndarray) -> np.ndarray:
    """
    This function predicts the stock prices using the LSTM model.

    Args:
        model:
            The LSTM model
        x_test:
            The testing input data

    Returns:
        The predicted stock prices
    """
    x_test = np.squeeze(x_test)

    predicted_prices = model.predict(x_test)

    predicted_prices = predicted_prices.reshape(-1, 1)

    predicted_prices = scaler.inverse_transform(predicted_prices)

    return predicted_prices


def test_xgboost_model(batch_size: int, num_epochs: int, x_train_list: list, n_estimators: int, max_depth: int,
                       grow_policy: str, colsample_bytree: float, reg_lambda: float, x_test_list: list,
                       lookback_period: int) -> dict:
    data = {}

    dataset_number = 1

    for x_train_original, x_test_original in zip(x_train_list, x_test_list):
        scaler, (x_train, y_train), (x_test, y_test) = (
            generate_training_and_testing_data(training_data=x_train_original, testing_data=x_test_original,
                                               lookback_period=lookback_period))

        rmse_list = []
        percentage_01_accuracy_list = []
        percentage_1_accuracy_list = []

        for i in range(3):
            print(f"Dataset {dataset_number} - Testing_Iteration {i + 1}")

            model = construct_xgboost_price_prediction_model(batch_size=batch_size, num_epochs=num_epochs,
                                                             x_train=x_train,
                                                             y_train=y_train, n_estimators=n_estimators,
                                                             max_depth=max_depth,
                                                             grow_policy=grow_policy, colsample_bytree=colsample_bytree,
                                                             reg_lambda=reg_lambda)

            print("Model trained")

            predicted_prices = predict_xgboost_prices(model=model, scaler=scaler, x_test=x_test)

            predicted_prices = np.squeeze(predicted_prices)

            y_test = x_test_original[lookback_period:]

            rmse = np.sqrt(np.mean(np.power((predicted_prices - y_test), 2)))

            predicted_prices = list(predicted_prices)

            if len(set(predicted_prices)) == 1:
                print("Values are same")

            percentage_01_accuracy = calculate_prediction_percentage_within_threshold(predicted_values=predicted_prices,
                                                                                      actual_values=y_test,
                                                                                      threshold_percentage=0.1)

            percentage_1_accuracy = calculate_prediction_percentage_within_threshold(predicted_values=predicted_prices,
                                                                                     actual_values=y_test,
                                                                                     threshold_percentage=1)

            rmse_list.append(rmse)
            percentage_01_accuracy_list.append(percentage_01_accuracy)
            percentage_1_accuracy_list.append(percentage_1_accuracy)

        rmse = np.mean(rmse_list)
        percentage_01_accuracy = np.mean(percentage_01_accuracy_list)
        percentage_1_accuracy = np.mean(percentage_1_accuracy_list)

        data[f"dataset_{dataset_number}"] = {"rmse": round(rmse, 2),
                                             "percentage_01_accuracy": round(percentage_01_accuracy, 2),
                                             "percentage_1_accuracy": round(percentage_1_accuracy, 2)}

        dataset_number += 1

    return data


def optimize_xgboost_model_random_search(batch_size_list: list[int], num_epochs: int, n_estimators_list: list[int],
                                         max_depth_list: list[int], grow_policy_list: list[str],
                                         colsample_bytree_list: list[float], reg_lambda_list: list[float],
                                         x_train_list: list, x_test_list: list,
                                         lookback_period_list: list[int],
                                         number_of_iterations: int) -> dict:
    data = {}

    cross_product = list(itertools.product(batch_size_list, n_estimators_list, max_depth_list, grow_policy_list,
                                           colsample_bytree_list, reg_lambda_list, lookback_period_list))

    selected_hyperparameters = random.sample(cross_product, number_of_iterations)

    iteration = 1

    for batch_size, n_estimators, max_depth, grow_policy, colsample_bytree, reg_lambda, lookback_period in selected_hyperparameters:
        print(f"Random Search Iteration {iteration}")
        print(f"Selected Hyperparameters: batch_size = {batch_size}, n_estimators = {n_estimators}, "
              f"max_depth = {max_depth}, grow_policy = {grow_policy}, colsample_bytree = {colsample_bytree}, "
              f"reg_lambda = {reg_lambda}, lookback_period = {lookback_period}")

        data[(("batch_size", batch_size), ("n_estimators", n_estimators), ("max_depth", max_depth),
              ("grow_policy", grow_policy), ("colsample_bytree", colsample_bytree), ("reg_lambda", reg_lambda),
              ("lookback_period", lookback_period))
        ] = test_xgboost_model(batch_size=batch_size, num_epochs=num_epochs,
                               x_train_list=x_train_list, n_estimators=n_estimators,
                               max_depth=max_depth, grow_policy=grow_policy,
                               colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
                               x_test_list=x_test_list,
                               lookback_period=lookback_period)

        print(data[(("batch_size", batch_size), ("n_estimators", n_estimators), ("max_depth", max_depth),
                    ("grow_policy", grow_policy), ("colsample_bytree", colsample_bytree), ("reg_lambda", reg_lambda),
                    ("lookback_period", lookback_period))
              ])

        iteration += 1

    return data
