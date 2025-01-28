#milestone 3 task 3, STEP 1A
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import boxcox, yeojohnson


class Plotter:
    """
    A class to visualize insights from a Pandas DataFrame.
    """  
    def __init__(self, df: pd.DataFrame, updated_df: pd.DataFrame = None):
        """
        Initializes the Plotter class.

        Args:
            original_df (pd.DataFrame): The DataFrame to visualize before processing.
            updated_df (pd.DataFrame, optional): The DataFrame to visualize after processing. Defaults to None.
        """
        self.df = df
        self.updated_df = updated_df


    def plot_nulls(self, before: bool = True) -> None:
        """
        Generates a bar plot of NULL values in each column before or after handling missing data.

        Args:
            before (bool): Whether the plot is for data before or after null handling. Defaults to True.
        """
        null_counts = self.df.isnull().sum()
        null_counts = null_counts[null_counts > 0]

        title = "NULL Values Before Handling" if before else "NULL Values After Handling"

        if null_counts.empty:
            print(f"No NULL values found in the DataFrame ({title}).")
            return

        plt.figure(figsize=(10, 6))
        sns.barplot(x=null_counts.index, y=null_counts.values)
        plt.xticks(rotation=45)
        plt.ylabel("Number of NULLs")
        plt.title(title)
        plt.show()



    def plot_hist(self, column):
        """
        Plots a histogram for a specific column.

        Args:
            column (str): Name of the column to plot.
        """
        # Check for NaNs or infinite values
        if self.df[column].isnull().any() or np.isinf(self.df[column]).any():
            print(f"Skipping column '{column}' due to NaNs or infinite values.")
            return

        # Ensure column has variance
        if self.df[column].nunique() <= 1:
            print(f"Skipping column '{column}' due to insufficient unique values.")
            return

        # Drop NaNs and infinite values
        column_data = self.df[column].dropna()
        column_data = column_data[np.isfinite(column_data)]

        # Plot histogram
        try:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=column_data, kde=True, bins=20)
            plt.title(f"Histogram for {column}")
            plt.show()
        except Exception as e:
            print(f"Error plotting histogram for column '{column}': {e}")

    def plot_box(self, column):
        """
        Plots a boxplot for a specific column.

        Args:
            column (str): Name of the column to plot.
        """
        # Similar checks as in plot_hist
        if self.df[column].isnull().any() or np.isinf(self.df[column]).any():
            print(f"Skipping column '{column}' due to NaNs or infinite values.")
            return

        if self.df[column].nunique() <= 1:
            print(f"Skipping column '{column}' due to insufficient unique values.")
            return

        try:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=self.df[column])
            plt.title(f"Boxplot for {column}")
            plt.show()
        except Exception as e:
            print(f"Error plotting boxplot for column '{column}': {e}")

    def plot_null_comparison(self) -> None:
        """
        Generates a bar plot to visualize the removal of NULL values
        by comparing the original and updated DataFrames.
        """
        original_nulls = self.df.isnull().sum()
        updated_nulls = self.updated_df.isnull().sum()

        # Combine the two sets of NULL counts for comparison
        comparison_df = pd.DataFrame({
            "Before": original_nulls,
            "After": updated_nulls
        })

        # Keep only columns that had NULL values originally
        comparison_df = comparison_df[comparison_df["Before"] > 0]

        if comparison_df.empty:
            print("No NULL values to visualize.")
            return

        # Create a bar plot for comparison
        comparison_df.plot(kind="bar", figsize=(12, 6), color=["red", "green"])
        plt.title("Comparison of NULL Values Before and After Handling")
        plt.ylabel("Number of NULLs")
        plt.xticks(rotation=45)
        plt.legend(["Before", "After"])
        plt.show()

    def plot_distributions(self, *columns):
        """
        Plots distributions for specified columns.

        Args:
            *columns: Column names to plot.
        """
        for column in columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.df[column].dropna(), kde=True)
            plt.title(f"Distribution of {column}")
            plt.show()
        

class DataFrameTransform:
    """
    A class to perform EDA transformations on a Pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataFrameTransform class with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to transform.
        """
        self.df = df

    def check_nulls(self) -> pd.DataFrame:
        """
        Checks the amount of NULL values in each column and their percentage.

        Returns:
            pd.DataFrame: A DataFrame summarizing NULL values and percentages.
        """
        null_counts = self.df.isnull().sum()
        null_percentages = (null_counts / len(self.df)) * 100
        null_summary = pd.DataFrame({
            "Null Count": null_counts,
            "Percentage": null_percentages
        })
        print("NULL Summary:")
        print(null_summary)
        return null_summary

    def drop_column(self, column_name: str="") -> None:
        """
        Drops a specific column from the DataFrame.

        Args:
            column_name (str): The name of the column to drop.
        """
        if column_name in self.df.columns:
            self.df.drop(columns=[column_name], inplace=True)
            print(f"Column '{column_name}' has been dropped.")
        else:
            print(f"Column '{column_name}' not found in the DataFrame.")

    def drop_columns_with_nulls(self, threshold: float = 50.0) -> None:
        """
        Drops columns with a percentage of NULL values exceeding the given threshold.

        Args:
            threshold (float): The percentage threshold for dropping columns. Defaults to 50.0.
        """
        null_percentages = (self.df.isnull().sum() / len(self.df)) * 100
        columns_to_drop = null_percentages[null_percentages > threshold].index
        self.df.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns with NULL values exceeding {threshold}%: {list(columns_to_drop)}")

    def impute_nulls(self, strategy: str = "mean") -> None:
        """
        Imputes NULL values in numeric columns using the specified strategy.

        Args:
            strategy (str): The imputation strategy ("mean" or "median"). Defaults to "mean".
        """
        numeric_columns = self.df.select_dtypes(include=["number"]).columns
        for column in numeric_columns:
            if self.df[column].isnull().sum() > 0:
                if strategy == "mean":
                    imputation_value = self.df[column].mean()
                elif strategy == "median":
                    imputation_value = self.df[column].median()
                else:
                    raise ValueError("Invalid strategy. Use 'mean' or 'median'.")
                self.df[column].fillna(imputation_value, inplace=True)
                print(f"Imputed NULL values in column '{column}' using {strategy}.")

    def impute_null(self, column_name, strategy="mean", fill_value=None):
        """
        Imputes null values in the specified column based on the chosen strategy.

        Parameters:
        - column_name (str): The column to impute.
        - strategy (str): The imputation strategy. Options are "mean", "median", "mode", or "constant".
        - fill_value: The value to use when strategy="constant". Default is None.
        """
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        if strategy == "mean":
            value = self.df[column_name].mean()
        elif strategy == "median":
            value = self.df[column_name].median()
        elif strategy == "mode":
            # Compute the mode; if multiple modes exist, use the first one
            value = self.df[column_name].mode()
            if value.empty:
                raise ValueError(f"Column '{column_name}' has no mode to impute.")
            value = value.iloc[0]
        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("For strategy='constant', a fill_value must be provided.")
            value = fill_value
        else:
            raise ValueError(f"Invalid strategy '{strategy}'. Choose from 'mean', 'median', 'mode', or 'constant'.")

        # Impute the null values
        self.df[column_name].fillna(value, inplace=True)

    def drop_rows_with_nulls(self) -> None:
        """
        Drops rows with any NULL values from the DataFrame.
        """
        initial_shape = self.df.shape
        self.df.dropna(inplace=True)
        final_shape = self.df.shape
        print(f"Dropped rows with NULL values. Rows before: {initial_shape[0]}, Rows after: {final_shape[0]}.")

    def detect_outliers(self, column: str, method: str = "IQR") -> pd.Series:
        """
        Identifies outliers in a column using the specified method.

        Args:
            column (str): The column to analyze for outliers.
            method (str): The method to use for outlier detection ("IQR" or "Z-Score"). Defaults to "IQR".

        Returns:
            pd.Series: A boolean Series indicating outliers (True for outliers, False otherwise).
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        if method == "IQR":
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        elif method == "Z-Score":
            mean = self.df[column].mean()
            std_dev = self.df[column].std()
            z_scores = (self.df[column] - mean) / std_dev
            outliers = (z_scores < -3) | (z_scores > 3)
        else:
            raise ValueError("Invalid method. Use 'IQR' or 'Z-Score'.")

        print(f"Outliers detected in column '{column}': {outliers.sum()} out of {len(self.df)} rows.")
        return outliers

    def remove_outliers(self, column: str, method: str = "IQR") -> None:
        """
        Removes rows containing outliers in the specified column.

        Args:
            column (str): The column to analyze and remove outliers from.
            method (str): The method to use for outlier detection ("IQR" or "Z-Score"). Defaults to "IQR".
        """
        outliers = self.detect_outliers(column, method=method)
        initial_shape = self.df.shape
        self.df = self.df[~outliers]
        final_shape = self.df.shape

        print(f"Removed outliers from column '{column}'.")
        print(f"Rows before: {initial_shape[0]}, Rows after: {final_shape[0]}.")

    def transform_outliers0(self, column: str, method: str = "IQR", strategy: str = "cap") -> None:
        """
        Transforms outliers in a column by capping or flooring them.

        Args:
            column (str): The column to analyze and transform outliers.
            method (str): The method to use for outlier detection ("IQR" or "Z-Score"). Defaults to "IQR".
            strategy (str): The strategy to use for transformation ("cap" or "floor"). Defaults to "cap".
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        if method == "IQR":
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == "Z-Score":
            mean = self.df[column].mean()
            std_dev = self.df[column].std()
            lower_bound = mean - 3 * std_dev
            upper_bound = mean + 3 * std_dev
        else:
            raise ValueError("Invalid method. Use 'IQR' or 'Z-Score'.")

        if strategy == "cap":
            self.df[column] = np.where(self.df[column] > upper_bound, upper_bound, self.df[column])
            self.df[column] = np.where(self.df[column] < lower_bound, lower_bound, self.df[column])
        elif strategy == "floor":
            self.df[column] = np.where(self.df[column] > upper_bound, lower_bound, self.df[column])
        else:
            raise ValueError("Invalid strategy. Use 'cap' or 'floor'.")

        print(f"Outliers in column '{column}' transformed using strategy '{strategy}'.")

    def identify_skewed_columns(self, threshold: float = 1.0):
        """
        Identifies columns with skewness above a given threshold.

        Args:
            threshold (float): Skewness threshold for identifying skewed columns.

        Returns:
            list: List of skewed columns.
        """
        skewness = self.df.skew(numeric_only=True)
        skewed_columns = skewness[abs(skewness) > threshold].index.tolist()
        print(f"Skewed columns (|skewness| > {threshold}): {skewed_columns}")
        return skewed_columns

    def transform_skew(self, column, method='log'):
        """
        Applies a transformation to reduce skewness in a column.

        Args:
            column (str): The column to transform.
            method (str): The transformation method ('log', 'sqrt', 'boxcox', 'yeojohnson').

        Returns:
            None
        """
        try:
            # Ensure column is a float type to avoid type-casting issues
            self.df[column] = self.df[column].astype(float)
            
            # Apply the chosen transformation
            if method == 'log':
                self.df[column] = np.log1p(self.df[column].clip(lower=1e-6))
            elif method == 'sqrt':
                self.df[column] = np.sqrt(self.df[column].clip(lower=0))
            elif method == 'boxcox':
                self.df[column], _ = boxcox(self.df[column].clip(lower=1e-6))
            elif method == 'yeojohnson':
                self.df[column], _ = yeojohnson(self.df[column])
            else:
                raise ValueError(f"Unsupported transformation method: {method}")
            
            print(f"Applied {method} transformation to column '{column}'.")
        except Exception as e:
            print(f"Error applying {method} transformation to column '{column}': {e}")
    
    def remove_outliers1(self, column, method="zscore", threshold=3):
        """
        Identifies and removes outliers from a column.

        Args:
            column (str): The column to process.
            method (str): The method to identify outliers ('zscore', 'iqr', or 'clip').
            threshold (float): Threshold for identifying outliers (used for z-score or IQR).

        Returns:
            None
        """
        try:
            if method == "zscore":
                z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
                outliers = z_scores > threshold
                self.df = self.df.loc[~outliers]
            elif method == "iqr":
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            elif method == "clip":
                lower_bound = self.df[column].quantile(0.01)
                upper_bound = self.df[column].quantile(0.99)
                self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)
            else:
                raise ValueError(f"Unsupported method: {method}")
            print(f"Outliers removed from column '{column}' using method '{method}'.")
        except Exception as e:
            print(f"Error processing column '{column}' for outliers: {e}")

    def remove_outliers(self, columns=None, method="zscore", threshold=3):
        """
        Identifies and removes or transforms outliers from specified columns in the DataFrame.

        Args:
            columns (list or None): List of column names to process. If None, all numeric columns will be processed.
            method (str): Method to identify outliers ('zscore', 'iqr', or 'clip').
            threshold (float): Threshold for identifying outliers (used for z-score or IQR).

        Returns:
            None: Modifies the DataFrame in place.
        """
        if columns is None:
            # Default to numeric columns if no columns are specified
            columns = self.df.select_dtypes(include=["float64", "int64"]).columns.tolist()

        for column in columns:
            try:
                if method == "zscore":
                    z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
                    outliers = z_scores > threshold
                    self.df = self.df.loc[~outliers]
                elif method == "iqr":
                    Q1 = self.df[column].quantile(0.25)
                    Q3 = self.df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
                elif method == "clip":
                    lower_bound = self.df[column].quantile(0.01)
                    upper_bound = self.df[column].quantile(0.99)
                    self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                print(f"Processed outliers in column '{column}' using method '{method}'.")
            except Exception as e:
                print(f"Error processing column '{column}': {e}")

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
    
    def drop_rows_with_nulls_in_column(self, column_name: str) -> None:
        """
        Drops rows with NULL values in the specified column.

        Args:
            column_name (str): The name of the column to check for NULL values.
        """
        if column_name in self.df.columns:
            initial_shape = self.df.shape
            self.df = self.df[self.df[column_name].notnull()]
            final_shape = self.df.shape
            print(f"Dropped rows with NULL values in column '{column_name}'. Rows before: {initial_shape[0]}, Rows after: {final_shape[0]}.")
        else:
            print(f"Column '{column_name}' not found in the DataFrame.")