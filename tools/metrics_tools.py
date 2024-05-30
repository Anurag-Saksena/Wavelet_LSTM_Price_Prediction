import math

from pandas import Timestamp

from tools.date_tools import get_next_trading_day, midnight, get_list_of_trading_days


def calculate_prediction_percentage_within_threshold(predicted_values: list[float], actual_values: list[float],
                                                     threshold_percentage: float) -> float:
    """Calculating the number of data points that were predicted correctly within a threshold percentage"""

    correct_predictions = 0
    for actual_data, predicted_data in zip(actual_values, predicted_values):
        price_difference_percentage = (abs(predicted_data - actual_data) / actual_data) * 100
        if price_difference_percentage <= threshold_percentage:
            correct_predictions += 1

    return round(correct_predictions * 100 / len(predicted_values), 2)


def calculate_consecutive_losses_and_gains(net_transaction_list: list[dict]) -> (int, float, int, float):
    """
    This function takes a net transaction list as input and returns a tuple of (maximum_number_of_consecutive_gains,
    maximum_consecutive_gain, maximum_number_of_consecutive_losses, maximum_consecutive_loss).

    Note: Remember even 1 gain is treated as 1 consecutive gain and 1 loss is treated as 1 consecutive loss

    Args:
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

    """
    list_of_consecutive_losses = []
    list_of_number_of_consecutive_losses = []
    number_of_consecutive_losses = 0
    consecutive_loss = 0

    list_of_consecutive_gains = []
    list_of_number_of_consecutive_gains = []
    number_of_consecutive_gains = 0
    consecutive_gain = 0

    for net_transaction in net_transaction_list:
        if net_transaction["result"] == "loss":
            list_of_consecutive_gains.append(consecutive_gain)
            list_of_number_of_consecutive_gains.append(number_of_consecutive_gains)
            number_of_consecutive_gains = 0
            consecutive_gain = 0

            number_of_consecutive_losses += 1
            consecutive_loss += net_transaction["profit_or_loss"]

        else:
            list_of_consecutive_losses.append(consecutive_loss)
            list_of_number_of_consecutive_losses.append(number_of_consecutive_losses)
            number_of_consecutive_losses = 0
            consecutive_loss = 0

            number_of_consecutive_gains += 1
            consecutive_gain += net_transaction["profit_or_loss"]

    # Filling in the values for the last profit or loss. Eg. If there were just 5 profits, then the values of
    # 'consecutive_gain' and 'number_of_consecutive_gains' would never be added to their lists because the first if
    # clause wouldn't execute.
    if consecutive_gain != 0:
        list_of_consecutive_gains.append(consecutive_gain)
    if number_of_consecutive_gains != 0:
        list_of_number_of_consecutive_gains.append(number_of_consecutive_gains)
    if consecutive_loss != 0:
        list_of_consecutive_losses.append(consecutive_loss)
    if number_of_consecutive_losses != 0:
        list_of_number_of_consecutive_losses.append(number_of_consecutive_losses)

    maximum_number_of_consecutive_gains = \
        max(list_of_number_of_consecutive_gains) if list_of_number_of_consecutive_losses else 0

    maximum_consecutive_gain = max(list_of_consecutive_gains) if list_of_consecutive_gains else 0

    maximum_number_of_consecutive_losses = \
        max(list_of_number_of_consecutive_losses) if list_of_number_of_consecutive_losses else 0

    maximum_consecutive_loss = min(list_of_consecutive_losses) if list_of_consecutive_losses else 0

    return (maximum_number_of_consecutive_gains, maximum_consecutive_gain, maximum_number_of_consecutive_losses,
            maximum_consecutive_loss)


def calculate_max_drawdown(net_funds_list: list[float]) -> float:
    """
    Calculating the max drawdown as a percentage. The max drawdown percentage is defined as the maximum percentage
    the value of a given portfolio ('net_funds') falls from its previous peak.

    See unit tests for examples.

    Args:
        net_funds_list:
            The 'net_funds_list' contains a list of the net fund values from the start of the strategy to the end of the
            strategy with values stored at the beginning of the strategy and at the end of each net transaction.

    """

    peak = net_funds_list[0]
    max_drawdown = 0

    for value in net_funds_list:
        if value > peak:
            peak = value
        else:
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return round(max_drawdown, 4)


def calculate_pnl_statistics(net_pnl_list: list[float], profits_list: list[float], losses_list: list[float]) -> tuple[
    float, float, float, float, float, int, int, float, float, float, float]:
    """
    Calculates and returns various statistics related to profit and loss (PNL) based on the provided input lists as a
    tuple (net_profit, net_loss, r_factor, largest_profit, largest_loss, number_of_profits, number_of_losses,
    average_gain, average_loss, net_pnl, average_net_profit_or_loss)

    Remember that len(net_pnl_list) == len(profits_list) + len(losses_list)

    Args:
        net_pnl_list:
            A list of net PNL values over time.
        profits_list:
            A list of individual profit values.
        losses_list:
            A list of individual loss values. Remember loss values are negative numbers or 0.

    Returns:
        tuple: A tuple of the following statistics:
            - net_profit (float): Total net profit.
            - net_loss (float): Total net loss.
            - r_factor (float): R-factor, calculated as the absolute value of net_profit divided by net_loss
                                and math.inf if net_loss is 0
            - largest_profit (float): Largest individual profit value.
            - largest_loss (float): Largest individual loss value (in terms of magnitude). (This value is negative)
            - number_of_profits (int): Total number of profits.
            - number_of_losses (int): Total number of losses.
            - average_gain (float): Average profit value.
            - average_loss (float): Average loss value.
            - net_pnl (float): Net PNL at the end of the period.
            - average_net_profit_or_loss (float): Average net profit/loss per net transaction.

    """
    net_pnl = net_pnl_list[-1] if net_pnl_list else 0

    if profits_list:
        net_profit = sum(profits_list)
        largest_profit = max(profits_list)
        number_of_profits = len(profits_list)
    else:
        net_profit = 0
        largest_profit = 0
        number_of_profits = 0

    if losses_list:
        net_loss = sum(losses_list)
        largest_loss = min(losses_list)
        number_of_losses = len(losses_list)
    else:
        net_loss = 0
        largest_loss = 0
        number_of_losses = 0

    net_profit = round(net_profit, 2)
    net_loss = round(net_loss, 2)
    largest_profit = round(largest_profit, 2)
    largest_loss = round(largest_loss, 2)

    r_factor = round(abs(net_profit / net_loss), 2) if net_loss else math.inf

    average_gain = round(net_profit / number_of_profits, 2) if number_of_profits else 0

    average_loss = round(net_loss / number_of_losses, 2) if number_of_losses else 0

    number_of_net_transactions = number_of_profits + number_of_losses

    average_net_profit_or_loss = round(net_pnl / number_of_net_transactions, 2) if number_of_net_transactions else 0

    return (net_profit, net_loss, r_factor, largest_profit, largest_loss, number_of_profits, number_of_losses,
            average_gain, average_loss, net_pnl, average_net_profit_or_loss)


def calculate_win_and_loss_statistics(number_of_profits: int, number_of_losses: int) -> tuple[float, float]:
    """
    Calculates win-to-loss ratio and hit rate based on the number of profits and losses and returns a tuple
    (win_to_loss_ratio, hit_rate)

    Args:
        number_of_profits (int):
            Number of profits.
        number_of_losses (int):
            Number of losses.

    Returns:
        tuple: A tuple containing the win-to-loss ratio and hit rate.
            - win_to_loss_ratio (float): Ratio of number of profits to number of losses. If number of losses is 0,
                                         this value is math.inf
            - hit_rate (float): Ratio of profitable net transactions to total net transactions. If number of net
                                transactions is 0, this value is math.inf

    """
    number_of_net_transactions = number_of_profits + number_of_losses

    win_to_loss_ratio = round(number_of_profits / number_of_losses, 2) if number_of_losses else math.inf

    hit_rate = round(number_of_profits / number_of_net_transactions, 2) if number_of_net_transactions else math.inf

    return win_to_loss_ratio, hit_rate


def calculate_roi_statistics(initial_funds: float, net_pnl: float,
                             net_pnl_list: list[float]) -> tuple[float, float, float]:
    """
    Calculates ROI (Return on Investment), maximum intermediate ROI, and minimum intermediate ROI based on the provided
    inputs. Maximum intermediate ROI is the maximum value the ROI reaches at any point during the backtest when
    calculated with minimum intermediate ROI being defined similarly.

    Args:
        initial_funds (float): Initial value of net funds
        net_pnl (float): Net profit or loss at the end of the backtest
        net_pnl_list (list[float]): List of net profit or loss values at the end of each net transaction. Remember that
                                    net profit or loss at the end of a net transaction is the sum of the profits or
                                    losses for all the net transactions upto that point.

    Returns:
        tuple[float, float, float]: A tuple containing the following statistics:
            - roi (float): Return on Investment as a percentage from 0 to 100
            - maximum_intermediate_roi (float): Maximum intermediate ROI as a percentage from 0 to 100
            - minimum_intermediate_roi (float): Minimum intermediate ROI as a percentage from 0 to 100

    """
    roi = round(net_pnl * 100 / initial_funds, 2)

    if net_pnl_list:
        max_net_pnl = max(net_pnl_list)
        min_net_pnl = min(net_pnl_list)
    else:
        max_net_pnl = 0
        min_net_pnl = 0

    maximum_intermediate_roi = round(max_net_pnl * 100 / initial_funds, 2)
    minimum_intermediate_roi = round(min_net_pnl * 100 / initial_funds, 2)

    return roi, maximum_intermediate_roi, minimum_intermediate_roi


def calculate_daily_strategy_summary(start_day: Timestamp, end_day: Timestamp,
                                     net_transaction_list: list[dict]) -> dict:
    """
    Calculates the daily strategy summary by computing the daily PNL (profit or loss) for each trading day
    between the start day and end day (both inclusive) and returns this information as a dictionary.

    Args:
        start_day:
            A Timestamp representing the start day at midnight.
        end_day:
            A Timestamp representing the end day at midnight
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

    Returns:
        dict: A dictionary containing the daily PNL for each trading day, where the keys are formatted as
        "DayOfWeek DD-Mon-YYYY" and the values are the corresponding daily PNL values.
        Eg. {"Friday 09-Jun-2023": 6000, "Monday 12-Jun-2023": -2000}

    """
    strategy_day_by_day_dict = {}

    trading_days = get_list_of_trading_days(start_day, end_day)

    for day in trading_days:
        day_pnl = 0

        for net_transaction in net_transaction_list:
            if midnight(net_transaction["buy_time"]) == day:
                day_pnl += net_transaction["profit_or_loss"]

        strategy_day_by_day_dict[f"{start_day.strftime('%A %d-%b-%Y')}"] = round(day_pnl, 2)

        start_day = get_next_trading_day(start_day)

    return strategy_day_by_day_dict


# In the context of a trading model, average profit refers to the average monetary gain per profitable trade executed by the trading model. It indicates how right the agent is, whenever it is right. Mathematically, the average profit can be calculated as:

# In the context of a trading model, average loss refers to the average monetary loss per losing trade executed by the trading model. It indicates how wrong the agent is, whenever it is wrong. Mathematically, the average loss can be calculated as: