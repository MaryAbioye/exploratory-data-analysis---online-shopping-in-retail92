import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    """
    A class to visualize insights from a Pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the Plotter class with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to visualize.
        """
        self.df = df

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

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the Plotter class with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to visualize.
        """
        self.df = df

    def plot_box(self, column: str) -> None:
        """
        Generates a box plot for the specified column to visualize outliers.

        Args:
            column (str): The column to visualize with a box plot.
        """
        if column not in self.df.columns:
            print(f"Column '{column}' not found in the DataFrame.")
            return

        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.df, x=column)
        plt.title(f"Box Plot for {column}")
        plt.show()

    def plot_hist(self, column: str) -> None:
        """
        Generates a histogram for the specified column to visualize distribution.

        Args:
            column (str): The column to visualize with a histogram.
        """
        if column not in self.df.columns:
            print(f"Column '{column}' not found in the DataFrame.")
            return

        plt.figure(figsize=(8, 6))
        sns.histplot(data=self.df, x=column, kde=True, bins=20)
        plt.title(f"Histogram for {column}")
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

    def transform_outliers(self, column: str, method: str = "IQR", strategy: str = "cap") -> None:
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

# Example Usage
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        "numeric_column": [10, 20, None, 40, 50],
        "category_column": ["A", None, "A", "C", "B"],
        "datetime_column": [pd.NaT, pd.Timestamp("2023-01-02"), None, pd.Timestamp("2023-01-04"), pd.NaT],
    }
    df = pd.DataFrame(data)

    print("Initial DataFrame:")
    print(df)
    print()

    # Step 1: Check NULLs and visualize
    transformer = DataFrameTransform(df)
    plotter = Plotter(df)

    plotter.plot_nulls(before=True)
    transformer.check_nulls()

    # Step 2: Drop columns with high NULL percentage (e.g., >50%)
    transformer.drop_columns_with_nulls(threshold=50)

    # Step 3: Impute NULLs in numeric columns
    transformer.impute_nulls(strategy="median")

    # Step 4: Drop remaining rows with NULLs
    transformer.drop_rows_with_nulls()

    # Check NULLs and visualize again
    transformer.check_nulls()
    plotter.plot_nulls(before=False)

    # Final DataFrame
    print("Final DataFrame:")
    print(transformer.df)
