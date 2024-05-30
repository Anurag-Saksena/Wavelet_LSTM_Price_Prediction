from typing import Optional

import finplot as fplt
from matplotlib import pyplot as plt
from pandas import Timestamp


def plot_line_chart(x_values: list[Timestamp], y_values: list[list[float]], y_labels: list[str],
                    circle_positions: Optional[list[list[int]]], colors: list[str]):
    """
    This function uses finplot to plot a line chart with the given x and y values, y labels, and colors. It is less
    customizable than the plot_line_chart_matplotlib() function but provides a more intuitive interactive interface.

    Example:
        df_train = load_time_series_csv_to_df("data/test/NIFTY_50__10minute__test.csv")

        circle_positions = [([0] * len(df_train["date"])) for _ in range(2)]
        circle_positions[0][-2] = -1
        circle_positions[0][-4] = 1
        circle_positions[1][0] = True

        chart = plot_line_chart(x_values=df_train["date"], y_values=[df_train["open"], df_train["close"]],
                                y_labels=["Open", "Close"], circle_positions=circle_positions,
                                colors=["r", "g"])
    """
    # w = fplt.foreground = '#eef'
    # b = fplt.background = fplt.odd_plot_background = '#242320'
    # fplt.candle_bull_color = fplt.volume_bull_color = fplt.candle_bull_body_color = fplt.volume_bull_body_color = '#352'

    fplt.foreground = '#FFFFFF'
    fplt.background = '#FFFFFF'

    ax1 = fplt.create_plot(rows=1)
    # fplt.background = '#ff0'
    fplt.background = '#FFFFFF'
    # print("\n\n\n\n\n\n\n\n\\n\n")

    for y_value, y_label, color in zip(y_values, y_labels, colors):
        fplt.plot(x_values, y_value, color=color, legend=y_label, ax=ax1)

    if circle_positions is not None:
        if len(circle_positions) != len(y_values):
            raise ValueError(
                "The length of the circle_positions list should be equal to the length of the y_values list")
        if len(circle_positions[0]) != len(y_values[0]):
            raise ValueError(
                "The length of the inner lists of the circle_positions list should be equal to the length of "
                "the inner lists of the y_values list")

        for inner_list_index, inner_list in enumerate(circle_positions):
            for index, value in enumerate(inner_list):
                if value != 0:
                    pos = fplt.pg.Point(index - 0.25, y_values[inner_list_index][index] - 0.75)
                    size = fplt.pg.Point(0.5, 2)
                    color = "r" if value == -1 else "g"
                    pen = fplt.pg.mkPen(color)
                    ellipse = fplt.pg.EllipseROI(pos, size, pen=pen, movable=False)
                    ax1.vb.addItem(ellipse)

    fplt.background = '#FFFFFF'
    print(fplt.foreground)
    fplt.foreground = '#FFFFFF'
    fplt.show()


def plot_line_chart_matplotlib(x_values: list, y_values: list[list], x_label: str, y_labels: list[str],
                               labels: list[str], circle_positions: list[list[bool]],
                               title: str, colors: tuple[str] = ('red', 'green'),
                               markers: tuple[str] = ('o', 's'), linestyles: tuple[str] = ('-', '-'),
                               linewidths: tuple[float] = (0.5, 0.5), markersizes: tuple[int] = (0, 0),
                               grid: bool = True,
                               legend: bool = True, save: bool = False, save_path: Optional[str] = None,
                               show: bool = True):
    """
    This function offers more customizable plotting using matplotlib and provides a background grid in plotted charts.

    Plots a line chart with the given x and y values, x and y labels, title, colors, markers, linestyles, linewidths,
    markersizes, labels, grid, legend, save, save_path, and show

    Args:
        x_values:
            The x values for the line chart
        y_values:
            The y values for the line chart. This should be a nested list, where each inner list represents a separate
            set of y values.
        x_label:
            The label for the x axis
        y_labels:
            A list of labels for the y axes corresponding to the multiple y values
        title:
            The title of the line chart
        colors:
            A list of colors for each line in the chart
        markers:
            A list of markers for each line in the chart
        linestyles:
            A list of line styles for each line in the chart
        linewidths:
            A list of line widths for each line in the chart
        markersizes:
            A list of marker sizes for each line in the chart
        labels:
            A list of labels for each line in the chart
        grid:
            If the parameter 'grid' is set to 'True', the grid will be displayed
        legend:
            If the parameter 'legend' is set to 'True', the legend will be displayed
        save:
            If the parameter 'save' is set to 'True', the line chart will be saved
        save_path:
            The path where the line chart will be saved

    Examples:
        plot_line_chart_matplotlib(x_values=df_train["date"], y_values=[df_train["open"], df_train["close"]], x_label="X-axis",
                y_labels=["Y1-axis", "Y2-axis"], title="Multiple Line Chart", colors=['red', 'green'],
                markers=['o', 's'], linestyles=['-', '-'], linewidths=[0.5, 0.5], markersizes=[0, 0],
                labels=["Line 1", "Line 2"], grid=True, legend=True, save=False, save_path="multiple_line_chart.png",
                show=True)

    """
    fig, ax = plt.subplots()

    for i, y_data in enumerate(y_values):
        ax.plot(x_values, y_data, color=colors[i], marker=markers[i], linestyle=linestyles[i], linewidth=linewidths[i],
                markersize=markersizes[i], label=labels[i])

    if circle_positions is not None:
        if len(circle_positions) != len(y_values):
            raise ValueError(
                "The length of the circle_positions list should be equal to the length of the y_values list")
        if len(circle_positions[0]) != len(y_values[0]):
            raise ValueError(
                "The length of the inner lists of the circle_positions list should be equal to the length of "
                "the inner lists of the y_values list")

        for inner_list_index, inner_list in enumerate(circle_positions):
            for index, value in enumerate(inner_list):
                if value == True:
                    cir = plt.Circle((x_values[index], y_values[inner_list_index][index]), 0.07, color='r', fill=False)
                    ax.set_aspect('equal', adjustable='datalim')
                    ax.add_patch(cir)

    plt.xlabel(x_label)

    # Set individual y-labels for each set of y values
    for i, y_label in enumerate(y_labels):
        plt.ylabel(y_label)

    plt.title(title)

    plt.xticks(rotation=45)

    if grid:
        ax.grid()
    if legend:
        ax.legend()
    if save:
        plt.savefig(save_path)
    if show:
        plt.show()

    return plt
