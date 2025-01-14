import yaml
from sqlalchemy import create_engine
import pandas as pd


class RDSDatabaseConnector:
    """
    A class to manage connections and data extraction from an RDS database.
    """

    def __init__(self, credentials: dict):
        """
        Initializes the RDSDatabaseConnector with database credentials.
        
        Args:
            credentials (dict): A dictionary containing the database credentials.
        """
        self.credentials = credentials
        self.engine = None

    def init_db_engine(self) -> None:
        """
        Initializes the SQLAlchemy engine using provided credentials.
        """
        try:
            self.engine = create_engine(
                f"postgresql://{self.credentials['RDS_USER']}:"
                f"{self.credentials['RDS_PASSWORD']}@"
                f"{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/"
                f"{self.credentials['RDS_DATABASE']}"
            )
            print("Database engine successfully initialized.")
        except Exception as e:
            print(f"Error initializing database engine: {e}")

    def extract_table_to_dataframe(self, table_name: str) -> pd.DataFrame:
        """
        Extracts data from a specified table into a Pandas DataFrame.

        Args:
            table_name (str): Name of the table to extract data from.

        Returns:
            pd.DataFrame: DataFrame containing the table data.
        """
        if not self.engine:
            raise ValueError("Database engine is not initialized. Call init_db_engine first.")

        try:
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql(query, self.engine)
            print(f"Data successfully extracted from table: {table_name}")
            return df
        except Exception as e:
            print(f"Error extracting data from table {table_name}: {e}")
            return pd.DataFrame()

    def save_dataframe_to_csv(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Saves a DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            file_path (str): Path to save the CSV file.
        """
        try:
            df.to_csv(file_path, index=False)
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving DataFrame to CSV: {e}")

    @staticmethod
    def load_csv_to_dataframe(file_path: str) -> pd.DataFrame:
        """
        Loads data from a CSV file into a Pandas DataFrame.

        Args:
            file_path (str): Path of the CSV file to load.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data successfully loaded from {file_path}")
            print(f"Data Shape: {df.shape}")
            print(f"Data Sample:\n{df.head()}")
            return df
        except Exception as e:
            print(f"Error loading CSV from {file_path}: {e}")
            return pd.DataFrame()


def load_db_credentials(file_path: str) -> dict:
    """
    Loads database credentials from a YAML file.

    Args:
        file_path (str): Path to the YAML file containing credentials.

    Returns:
        dict: Dictionary containing database credentials.
    """
    try:
        with open(file_path, 'r') as file:
            credentials = yaml.safe_load(file)
        print(f"Credentials successfully loaded from {file_path}")
        return credentials
    except Exception as e:
        print(f"Error loading credentials from {file_path}: {e}")
        return {}


if __name__ == "__main__":
    # Example Usage (not required in the actual code base)
    credentials_path = "credentials.yaml"
    credentials = load_db_credentials(credentials_path)

    connector = RDSDatabaseConnector(credentials)
    connector.init_db_engine()

    table_name = "customer_activity"
    df = connector.extract_table_to_dataframe(table_name)

    csv_file_path = "customer_activity_data.csv"
    connector.save_dataframe_to_csv(df, csv_file_path)

    loaded_df = RDSDatabaseConnector.load_csv_to_dataframe(csv_file_path)