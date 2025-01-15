import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DataCorrelationHandler:
    """
    A class to identify and remove highly correlated columns in a Pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataCorrelationHandler class with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
        """
        self.df = df
        self.correlation_matrix = None

    def compute_correlation_matrix(self) -> None:
        """
        Computes the correlation matrix for numeric columns in the DataFrame.
        """
        self.correlation_matrix = self.df.corr()
        print("Correlation matrix computed.")

    def plot_correlation_matrix(self) -> None:
        """
        Visualizes the correlation matrix using a heatmap.
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix has not been computed. Run `compute_correlation_matrix` first.")

        plt.figure(figsize=(10, 8))
        sns.heatmap(self.correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Matrix Heatmap")
        plt.show()

    def identify_highly_correlated_columns(self, threshold: float = 0.8) -> list:
        """
        Identifies columns that are highly correlated with other columns.

        Args:
            threshold (float): The correlation threshold above which columns are considered highly correlated. Defaults to 0.8.

        Returns:
            list: A list of columns to drop based on high correlation.
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix has not been computed. Run `compute_correlation_matrix` first.")

        correlated_features = set()
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i):
                if abs(self.correlation_matrix.iloc[i, j]) > threshold:
                    colname = self.correlation_matrix.columns[i]
                    correlated_features.add(colname)

        print(f"Columns identified as highly correlated (threshold={threshold}): {correlated_features}")
        return list(correlated_features)

    def remove_highly_correlated_columns(self, columns_to_drop: list) -> None:
        """
        Removes highly correlated columns from the DataFrame.

        Args:
            columns_to_drop (list): The list of columns to drop.
        """
        initial_shape = self.df.shape
        self.df.drop(columns=columns_to_drop, inplace=True)
        final_shape = self.df.shape

        print(f"Removed {len(columns_to_drop)} highly correlated columns.")
        print(f"DataFrame shape before: {initial_shape}, after: {final_shape}.")


# Example Usage
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],
        "C": [5, 3, 2, 4, 1],
        "D": [10, 20, 30, 40, 50],
    }
    df = pd.DataFrame(data)

    print("Initial DataFrame:")
    print(df)
    print()

    # Initialize the correlation handler
    corr_handler = DataCorrelationHandler(df)

    # Step 1: Compute and visualize correlation matrix
    corr_handler.compute_correlation_matrix()
    corr_handler.plot_correlation_matrix()

    # Step 2: Identify highly correlated columns
    columns_to_drop = corr_handler.identify_highly_correlated_columns(threshold=0.8)

    # Step 3 & 4: Remove highly correlated columns
    corr_handler.remove_highly_correlated_columns(columns_to_drop)

    print("DataFrame after removing highly correlated columns:")
    print(corr_handler.df)
