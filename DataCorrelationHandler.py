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

   
    def compute_correlation_matrix(self):
        """
        Compute and visualize the correlation matrix.
        """
        correlation_matrix = self.df.corr()

        # Visualize the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

        return correlation_matrix

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

    
    def remove_highly_correlated(self, threshold=0.9):
        """
        Identify and remove highly correlated columns.

        Args:
            threshold (float): Correlation threshold above which columns are removed.

        Returns:
            list: Columns removed from the dataset.
        """
        correlation_matrix = self.df.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

        # Identify columns to drop
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

        # Drop columns
        self.df.drop(columns=to_drop, inplace=True)

        print(f"Removed columns due to high correlation (threshold > {threshold}): {to_drop}")
        return to_drop


