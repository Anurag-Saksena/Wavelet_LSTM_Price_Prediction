import math

import numpy as np
from pandas import Timestamp, DataFrame, Timedelta, Series

from tools.csv_tools import save_df_to_csv, load_time_series_csv_to_df
from tools.general_tools import print_string_to_text_file, print_string_to_text_file_append, \
    transaction_log, format_print
from tools.metrics_tools import (calculate_consecutive_losses_and_gains, calculate_max_drawdown,
                                 calculate_pnl_statistics, calculate_win_and_loss_statistics, calculate_roi_statistics,
                                 calculate_daily_strategy_summary, calculate_prediction_percentage_within_threshold)
from tools.plot_tools import plot_line_chart, plot_line_chart_matplotlib


class StrategyFailedError(Exception):
    """
    This error is called when a strategy fails because of insufficient funds.
    """
    pass


def get_baseline_accuracy(actual_closing_prices: Series, threshold_percentage: float) -> float:
    predicted_closing_prices = actual_closing_prices.shift(1)
    assert math.isnan(predicted_closing_prices[0])

    predicted_closing_prices[0] = actual_closing_prices[0]

    return calculate_prediction_percentage_within_threshold(predicted_closing_prices, actual_closing_prices,
                                                            threshold_percentage)


def plot_incorrect_predictions(dates: list[Timestamp], actual_closing_prices: list[float],
                               predicted_closing_prices: list[float], mark_incorrect: bool,
                               mark_incorrect_threshold_percentage: float):
    """
    Plots the actual closing prices against the predicted closing prices with an option to mark incorrect predictions.

    Args:
        dates (List[Timestamp]): A list of timestamps representing the dates for the closing prices.
        actual_closing_prices (List[float]): A list of actual closing prices.
        predicted_closing_prices (List[float]): A list of predicted closing prices.
        mark_incorrect (bool): A boolean indicating whether to mark incorrect predictions.
        mark_incorrect_threshold_percentage (float): The threshold percentage difference
            between actual and predicted prices to consider a prediction as incorrect.

    Returns:
        None: The function does not return anything but generates a plot.

    Notes:
        - If `mark_incorrect` is False, the function plots a line chart without marking incorrect predictions.
        - If `mark_incorrect` is True, incorrect predictions are marked on the plot.

    Example:
        df = load_time_series_csv_to_df("data/test/NIFTY_50__10minute__test.csv")
        df["predicted_close"] = df["close"] + np.random.normal(0, 10, len(df))
        plot_incorrect_predictions(dates=df["date"], actual_closing_prices=df["close"],
                                   predicted_closing_prices=df["predicted_close"], mark_incorrect=True,
                                   mark_incorrect_threshold_percentage=0.1)

    """
    if mark_incorrect == False:
        plot_line_chart(x_values=dates, y_values=[actual_closing_prices, predicted_closing_prices],
                        y_labels=["Actual Closing Prices", "Predicted Closing Prices"], circle_positions=None,
                        colors=["r", "g"])

    else:
        incorrect_positions = []
        for actual, predicted in zip(actual_closing_prices, predicted_closing_prices):
            price_difference_percentage = abs(actual - predicted) * 100 / actual
            if price_difference_percentage > mark_incorrect_threshold_percentage:
                incorrect_positions.append(-1)
            else:
                incorrect_positions.append(0)

        plot_line_chart(x_values=dates, y_values=[actual_closing_prices, predicted_closing_prices],
                        y_labels=["Actual Closing Prices", "Predicted Closing Prices"],
                        circle_positions=[incorrect_positions, incorrect_positions], colors=["r", "g"])


def plot_profits_and_losses_with_prices(dates: list[Timestamp], actual_opening_prices: list[float],
                                        actual_closing_prices: list[float], predicted_closing_prices: list[float],
                                        net_transactions: list[dict], just_losses: bool, just_profits: bool):
    if just_losses and just_profits:
        raise ValueError("Both 'just_losses' and 'just_profits' cannot be set to True at the same time")

    dates_to_index = {date: index for index, date in enumerate(dates)}

    buy_index_list = []
    sell_index_list = []

    actual_opening_marker_index_list = [0] * len(dates)
    actual_closing_marker_index_list = [0] * len(dates)

    for net_transaction in net_transactions:
        if just_losses:
            if net_transaction["result"] == "loss":
                if net_transaction["net_transaction_type"] == "LONG":
                    actual_opening_marker_index_list[dates_to_index[net_transaction["buy_time"]]] = 1
                    actual_closing_marker_index_list[dates_to_index[net_transaction["sell_time"]]] = -1
                else:
                    actual_opening_marker_index_list[dates_to_index[net_transaction["sell_time"]]] = 1
                    actual_closing_marker_index_list[dates_to_index[net_transaction["buy_time"]]] = -1

        elif just_profits:
            if net_transaction["result"] == "profit":
                if net_transaction["net_transaction_type"] == "LONG":
                    actual_opening_marker_index_list[dates_to_index[net_transaction["buy_time"]]] = 1
                    actual_closing_marker_index_list[dates_to_index[net_transaction["sell_time"]]] = -1
                else:
                    actual_opening_marker_index_list[dates_to_index[net_transaction["sell_time"]]] = 1
                    actual_closing_marker_index_list[dates_to_index[net_transaction["buy_time"]]] = -1

        else:
            if net_transaction["net_transaction_type"] == "LONG":
                actual_opening_marker_index_list[dates_to_index[net_transaction["buy_time"]]] = 1
                actual_closing_marker_index_list[dates_to_index[net_transaction["sell_time"]]] = -1
            else:
                actual_opening_marker_index_list[dates_to_index[net_transaction["sell_time"]]] = 1
                actual_closing_marker_index_list[dates_to_index[net_transaction["buy_time"]]] = -1

    plot_line_chart(x_values=dates, y_values=[actual_opening_prices, actual_closing_prices, predicted_closing_prices],
                    y_labels=["Actual Opening Prices", "Actual Closing Prices", "Predicted Closing Prices"],
                    circle_positions=[actual_opening_marker_index_list, actual_closing_marker_index_list,
                                      [0] * len(dates)], colors=["r", "g", "b"])


def price_prediction_to_trades(dates: list[Timestamp], actual_opening_prices: list[float],
                               actual_closing_prices: list[float], expected_closing_prices: list[float],
                               threshold_percentage: float, interval_int: int, market_close_time: Timestamp,
                               initial_funds: float) -> tuple[list[dict], list[dict]]:
    """
    This function takes the actual open prices and the expected closing prices and returns a list of trades based on the
    threshold percentage. If the expected closing price for the current interval is greater than the actual opening
    price for the current interval by the threshold percentage, then the trade is 'buy'. If the expected closing price
    for the current interval is less than the actual opening price for the current interval by the threshold percentage,
    then the trade is 'sell'. If the expected closing price for the current interval is within the threshold percentage
    of the actual opening price for the current interval, then the trade is 'hold'.
    """

    transactions = []
    net_transactions = []
    net_pnl = 0

    funds_for_trading = initial_funds // 2

    net_funds = initial_funds

    for date, actual_open, actual_close, expected_close in zip(dates, actual_opening_prices, actual_closing_prices,
                                                               expected_closing_prices):

        if expected_close > actual_open * (1 + threshold_percentage/100):
            quantity = funds_for_trading // actual_open

            buy_transaction_details = {"transaction_type": "BUY", "time": date, "price": actual_open,
                                       "quantity": quantity, "opening_order": True}

            sell_time = date + Timedelta(minutes=interval_int)
            market_close_time_for_date = date.replace(hour=market_close_time.hour, minute=market_close_time.minute)
            if sell_time > market_close_time_for_date:
                sell_time = market_close_time_for_date

            sell_transaction_details = {"transaction_type": "SELL", "time": sell_time, "price": actual_close,
                                        "quantity": quantity, "opening_order": False}

            pnl = (actual_close - actual_open) * quantity
            net_pnl += pnl

            net_funds += pnl

            transactions.append(buy_transaction_details)
            transactions.append(sell_transaction_details)

            net_transaction_details = {"net_transaction_type": "LONG", "buy_price": actual_open,
                                       "sell_price": actual_close, "quantity": quantity, "buy_time": date,
                                       "sell_time": sell_time, "holding_period": sell_time - date, "pnl": pnl,
                                       "net_pnl": net_pnl, "net_funds": net_funds}

            net_transactions.append(net_transaction_details)

            if net_funds < 0:
                raise StrategyFailedError("Insufficient funds for trading")


        elif expected_close < actual_open * (1 - threshold_percentage/100):
            quantity = funds_for_trading // actual_open

            sell_transaction_details = {"transaction_type": "SELL", "time": date, "price": actual_open,
                                        "quantity": quantity, "opening_order": True}

            buy_time = date + Timedelta(minutes=interval_int)
            market_close_time_for_date = date.replace(hour=market_close_time.hour, minute=market_close_time.minute)
            if buy_time > market_close_time_for_date:
                buy_time = market_close_time_for_date

            buy_transaction_details = {"transaction_type": "BUY", "time": buy_time, "price": actual_close,
                                       "quantity": quantity, "opening_order": False}

            pnl = (actual_open - actual_close) * quantity
            net_pnl += pnl

            net_funds += pnl

            transactions.append(sell_transaction_details)
            transactions.append(buy_transaction_details)

            net_transaction_details = {"net_transaction_type": "SHORT", "sell_price": actual_open,
                                       "buy_price": actual_close, "quantity": quantity, "sell_time": date,
                                       "buy_time": buy_time, "holding_period": buy_time - date, "pnl": pnl,
                                       "net_pnl": net_pnl, "net_funds": net_funds}

            net_transactions.append(net_transaction_details)

            if net_funds < 0:
                raise StrategyFailedError("Insufficient funds for trading")

    return transactions, net_transactions


def calculate_strategy_metrics(initial_funds: float, start_day: Timestamp, end_day: Timestamp, entry_description: str,
                               exit_description: str, transaction_list: list[dict], net_transaction_list: list[dict],
                               transactions_csv_file: str, net_transactions_csv_file: str,
                               strategy_summary_log_file: str,
                               strategy_summary_csv_file: str) -> dict:
    """
    This function takes the given inputs, returns a strategy summary dict (see 3a) and generates 4 files that summarize
    the information from a backtest:

    Reminder: None values in a dictionary get converted into math.nan values in a DataFrame.

    1. This function takes the input transaction list and converts it to a CSV file stored at the path
    'transactions_csv_file'. In this transactions CSV file, there are separate columns for 'time_till_sell_trigger' and
    'time_till_buy_trigger'. There are also separate columns for 'amount_spent_without_charges' and
    'amount_earned_without_charges'. In addition, there are separate columns for 'amount_spent' and
    'amount_earned'. For these 3 cases, only 1 of the parameters will have a valid value for any given transaction.
    Eg. In the converted DataFrame row, for a BUY transaction, the 3 columns 'time_till_sell_trigger',
    'amount_spent_without_charges' and 'amount_spent' will have valid values that were passed in the input transaction
    list while the 3 columns 'time_till_buy_trigger', 'amount_earned_without_charges' and 'amount_earned' will have
    math.nan values because these values were not passed in the input list item for this BUY transaction.

    2. This function takes the input net transaction list and converts it to a CSV file stored at the path
    'net_transactions_csv_file'.

    3. This function takes the input net transaction list and uses it to calculate 3 separate pieces of information:
        a. It calculates a strategy summary dictionary which contains information about the strategy metrics:
           "Entry Description", "Exit Description", "Initial Funds", "Maximum Number Of Consecutive Gains",
           "Maximum Consecutive Gain", "Maximum Number Of Consecutive Losses", "Maximum Consecutive Loss",
           "Maximum Drawdown", "Total Profit", "Total Loss", "R Factor (Total Profit / Total Loss)", "Largest Profit",
           "Largest Loss", "Average Gain", "Average Loss", "Final Net Profit or Loss", "Average Net Profit Or Loss",
           "Number Of Net Transactions", "Number Of Wins", "Number Of Losses", "Win To Loss Ratio", "Hit Rate", "ROI",
           "Maximum Intermediate ROI", "Minimum Intermediate ROI"
        b. It calculates a list of net transaction summary dictionaries where each net transaction is used to calculate
           a summary dictionary of the format {"instrument_token", "transaction_pair", "buy_time", "sell_time",
                                               "holding_time", "result", "sell_trigger", "profit_or_loss"}
        c. It calculates a daily strategy summary dictionary as created by calculate_daily_strategy_summary(). This
           dictionary contains the daily PnL values for each trading day during the strategy time period.

       It then prints all 3 of these pieces of information to the strategy summary log file stored at the path
       'strategy_summary_log_file'. See unit tests for examples.

    4. This function also converts the strategy summary dictionary calculated in 3a. to a CSV file with 1 row and it
    then stores this CSV file at the path 'strategy_summary_csv_file'.

    All paths are defined with respect to project path

    Args:
        initial_funds:
            A float representing the initial value of net funds for the strategy
        start_day:
            A Timestamp representing midnight on the start day of the strategy
        end_day:
            A Timestamp representing midnight on the end day of the strategy
        entry_description:
            A string representing a description of the entry criteria for net transactions
        exit_description:
            A string representing a description of the exit criteria for net transactions
        transaction_list:
            A list of dicts where each dict contains details of a BUY or SELL transaction and each dict looks similar
            to this:
            {"instrument_token", "time", "opening_order", "trigger", "holding_period", "product_type",
            "transaction_type", "order_type", "slippage_absolute_value", "average_price", "max_allocation_used",
            "quantity", "max_quantity", "quantity_set_to_max_quantity_because_exceeded", "quantity_freeze",
            "quantity_set_to_quantity_freeze_because_exceeded", "iceberg_enabled", "stop_loss", "target", "broker",
            "stop_loss_price", "target_price", "time_till_sell_trigger", "amount_spent_without_charges",
            "net_funds_without_charges", "amount_spent", "net_funds"}
        net_transaction_list:
            A list of dictionaries with each dictionary containing the details of a net LONG or SHORT transaction.
            Each net transaction dictionary looks like this:
            {"instrument_token", "transaction_pair", "holding_period", "product_type", "buy_time", "buy_trigger",
             "buy_order_type", "buy_average_price", "max_allocation_used", "quantity", "max_quantity",
             "quantity_set_to_max_quantity_because_exceeded", "quantity_freeze",
             "quantity_set_to_quantity_freeze_because_exceeded", "iceberg_enabled", "stop_loss", "target",
             "stop_loss_price", "target_price", "time_till_trigger", "sell_time", "sell_trigger", "sell_order_type",
             "sell_average_price", "holding_time", "net_funds_without_charges", "net_funds", "buy_side_slippage",
             "sell_side_slippage", "total_buy_and_sell_slippage_for_entire_quantity", "profit_or_loss_without_charges",
             "profit_or_loss_without_slippage_accounted_but_other_charges_included", "profit_or_loss", "result",
             "net_profit_or_loss"}
        transactions_csv_file:
            A string representing the CSV file path to store transaction details.
        net_transactions_csv_file:
            A string representing the CSV file path to store net transaction details.
        strategy_summary_log_file:
            A string representing the log file path to store strategy summary metrics.
        strategy_summary_csv_file:
            A string representing the CSV file path to store strategy summary details.

    """
    # Constructing the net transactions csv file
    net_transactions_list_df = DataFrame(net_transaction_list)
    save_df_to_csv(net_transactions_list_df, net_transactions_csv_file)

    # Constructing the transactions csv file
    transactions_list_df = DataFrame(transaction_list)
    save_df_to_csv(transactions_list_df, transactions_csv_file)

    # The 'net_funds_list' contains a list of the net fund values from the start of the strategy to the end of the
    # strategy with values stored at the beginning of the strategy and at the end of each net transaction.
    net_funds_list = [initial_funds]

    profit_or_loss_list = []

    net_transaction_summary_list = []
    net_pnl_list = []

    losses_list = []
    profits_list = []

    # Creating a transaction summary for each net transaction and calculating the values of each of these lists
    for net_transaction in net_transaction_list:
        transaction_summary = {"instrument_token": net_transaction["instrument_token"],
                               "transaction_pair": net_transaction["transaction_pair"],
                               "buy_time": net_transaction["buy_time"],
                               "sell_time": net_transaction["sell_time"],
                               "holding_time": net_transaction["holding_time"],
                               "result": net_transaction["result"],
                               "sell_trigger": net_transaction["sell_trigger"],
                               "profit_or_loss": net_transaction["profit_or_loss"]}
        net_transaction_summary_list.append(transaction_summary)
        net_funds_list.append(net_transaction["net_funds"])
        profit_or_loss_list.append(net_transaction["profit_or_loss"])
        net_pnl_list.append(net_transaction["net_profit_or_loss"])
        if net_transaction["result"] == "loss":
            losses_list.append(net_transaction["profit_or_loss"])
        else:
            profits_list.append(net_transaction["profit_or_loss"])

    # Calculating strategy metrics

    # Calculating consecutive gains and consecutive losses
    (maximum_number_of_consecutive_gains, maximum_consecutive_gain, maximum_number_of_consecutive_losses,
     maximum_consecutive_loss) = calculate_consecutive_losses_and_gains(net_transaction_list)

    # Calculating max drawdown
    max_drawdown = calculate_max_drawdown(net_funds_list)

    # Calculating profit and loss statistics
    (net_profit, net_loss, r_factor, largest_profit, largest_loss, number_of_profits, number_of_losses,
     average_gain, average_loss, net_pnl, average_net_profit_or_loss) \
        = calculate_pnl_statistics(net_pnl_list, profits_list, losses_list)

    # Calculating statistics about the number of wins and losses
    win_to_loss_ratio, hit_rate = calculate_win_and_loss_statistics(number_of_profits, number_of_losses)

    # Calculating ROI statistics
    roi, maximum_intermediate_roi, minimum_intermediate_roi = \
        calculate_roi_statistics(initial_funds, net_pnl, net_pnl_list)

    # Constructing the strategy summary dictionary
    strategy_summary_dict = {"Entry Description": entry_description, "Exit Description": exit_description,
                             "Initial Funds": initial_funds,
                             "Maximum Number Of Consecutive Gains": maximum_number_of_consecutive_gains,
                             "Maximum Consecutive Gain": maximum_consecutive_gain,
                             "Maximum Number Of Consecutive Losses": maximum_number_of_consecutive_losses,
                             "Maximum Consecutive Loss": maximum_consecutive_loss, "Maximum Drawdown": max_drawdown,
                             "Total Profit": net_profit, "Total Loss": net_loss,
                             "R Factor (Total Profit / Total Loss)": r_factor, "Largest Profit": largest_profit,
                             "Largest Loss": largest_loss, "Average Gain": average_gain, "Average Loss": average_loss,
                             "Final Net Profit or Loss": net_pnl,
                             "Average Net Profit Or Loss": average_net_profit_or_loss,
                             "Number Of Net Transactions": number_of_profits + number_of_losses,
                             "Number Of Wins": number_of_profits, "Number Of Losses": number_of_losses,
                             "Win To Loss Ratio": win_to_loss_ratio, "Hit Rate": hit_rate, "ROI": roi,
                             "Maximum Intermediate ROI": maximum_intermediate_roi,
                             "Minimum Intermediate ROI": minimum_intermediate_roi}

    # Calculating daily summary of total profit or loss
    strategy_day_by_day_dict = calculate_daily_strategy_summary(start_day, end_day, net_transaction_list)

    # Creating the strategy summary log file
    print_string_to_text_file("Strategy Summary Metrics:\n\n", strategy_summary_log_file)
    transaction_log(strategy_summary_dict, strategy_summary_log_file)

    print_string_to_text_file_append("\n\nNet Transactions Summary:\n\n", strategy_summary_log_file)
    for net_transaction_summary in net_transaction_summary_list:
        transaction_log(net_transaction_summary, strategy_summary_log_file)

    print_string_to_text_file_append("\n\nStrategy Summary Day By Day:\n\n", strategy_summary_log_file)
    transaction_log(strategy_day_by_day_dict, strategy_summary_log_file)

    # Creating the strategy summary csv file
    strategy_summary_df = DataFrame([strategy_summary_dict])
    save_df_to_csv(strategy_summary_df, strategy_summary_csv_file)

    return strategy_summary_dict


# df = load_time_series_csv_to_df("data/test/NIFTY_50__10minute__test.csv")
#
# actual_closing_prices = df["close"]
#
# predicted_closing_prices = actual_closing_prices.shift(1)
# assert math.isnan(predicted_closing_prices[0])
# predicted_closing_prices[0] = actual_closing_prices[0]
#
# print(get_baseline_accuracy(df["close"], 0.025))
#
# transaction_list, net_transaction_list = price_prediction_to_trades(dates=df["date"], actual_opening_prices=df["open"],
#                                                                     actual_closing_prices=df["close"],
#                                                                     expected_closing_prices=predicted_closing_prices,
#                                                                     threshold_percentage=0.1, interval_int=10,
#                                                                     market_close_time=Timestamp("23-Feb-2024 15:29"),
#                                                                     initial_funds=2_000_000)
#
# format_print(transaction_list)
# format_print(net_transaction_list)


# df = load_time_series_csv_to_df("data/train/NIFTY_50__60_minute__train1.csv")
# df["predicted_close"] = df["close"] + np.random.normal(0, 10, len(df))
# plot_incorrect_predictions(dates=df["date"], actual_closing_prices=df["close"],
#                            predicted_closing_prices=df["predicted_close"], mark_incorrect=True,
#                            mark_incorrect_threshold_percentage=0.1)