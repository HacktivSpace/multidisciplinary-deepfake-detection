import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.config import config

def plot_histogram(data, column, bins=30, output_dir=config.LOG_DIR, filename='histogram.png'):
    """
    Plotting and saving histogram of a specified column.
    :param data: DataFrame containing the data
    :param column: Column to plot the histogram for
    :param bins: Number of bins for the histogram
    :param output_dir: Directory to save the plot
    :param filename: Name of the output file
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=bins, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Histogram plot saved to {os.path.join(output_dir, filename)}")

def plot_scatter(data, x_column, y_column, output_dir=config.LOG_DIR, filename='scatter_plot.png'):
    """
    Plotting and saving scatter plot of two specified columns.
    :param data: DataFrame containing the data
    :param x_column: Column to plot on the x-axis
    :param y_column: Column to plot on the y-axis
    :param output_dir: Directory to save the plot
    :param filename: Name of the output file
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[x_column], y=data[y_column])
    plt.title(f'Scatter Plot of {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Scatter plot saved to {os.path.join(output_dir, filename)}")

def plot_correlation_matrix(data, output_dir=config.LOG_DIR, filename='correlation_matrix.png'):
    """
    Plotting and saving correlation matrix of the data.
    :param data: DataFrame containing the data
    :param output_dir: Directory to save the plot
    :param filename: Name of the output file
    """
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Correlation matrix plot saved to {os.path.join(output_dir, filename)}")

def plot_time_series(data, date_column, value_column, output_dir=config.LOG_DIR, filename='time_series.png'):
    """
    Plotting and saving time series plot.
    :param data: DataFrame containing the data
    :param date_column: Column containing the date values
    :param value_column: Column containing the values to plot
    :param output_dir: Directory to save the plot
    :param filename: Name of the output file
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data[date_column], data[value_column])
    plt.title(f'Time Series of {value_column} over Time')
    plt.xlabel('Date')
    plt.ylabel(value_column)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Time series plot saved to {os.path.join(output_dir, filename)}")

if __name__ == "__main__":

    example_data = pd.DataFrame({
        'date': pd.date_range(start='2021-01-01', periods=100, freq='D'),
        'value': np.random.randn(100).cumsum(),
        'category': np.random.choice(['A', 'B', 'C'], size=100),
        'value2': np.random.randn(100)
    })

    plot_histogram(example_data, 'value', filename='example_histogram.png')

    plot_scatter(example_data, 'value', 'value2', filename='example_scatter_plot.png')

    plot_correlation_matrix(example_data, filename='example_correlation_matrix.png')

    plot_time_series(example_data, 'date', 'value', filename='example_time_series.png')
