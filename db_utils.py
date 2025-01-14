class RDSDatabaseConnector:
    def load_credentials(file_path='credentials.yaml'):
    # Load the credentials from a YAML file and return them as a dictionary.
        try:
            with open(file_path, 'r') as file:
                credentials = yaml.safe_load(file)
                return credentials
        except Exception as e:
            print(f"Error loading credentials: {e}")
            raise

    def __init__(self, credentials):
        """
        Initialize the RDSDatabaseConnector instance with connection parameters from the credentials dictionary.
        
        :param credentials: A dictionary containing the database connection details.
        """
        print("Jesse")
        self.host = credentials['RDS_HOST']
        self.port = credentials['RDS_PORT']
        self.database = credentials['RDS_DATABASE']
        self.user = credentials['RDS_USER']
        self.password = credentials['RDS_PASSWORD']
        self.engine = None

    def initialize_engine(self):
        """
        Initialize the SQLAlchemy engine to connect to the RDS database using the credentials.
        """
        try:
            # Create the connection string for SQLAlchemy
            connection_string = f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'
            
            # Initialize the SQLAlchemy engine
            self.engine = create_engine(connection_string)
            print("SQLAlchemy engine initialized successfully.")
        except Exception as e:
            print(f"Error initializing SQLAlchemy engine: {e}")
            raise
    
    def extract_data(self):
        """
        Extract data from the 'customer_activity' table and return it as a Pandas DataFrame.
        
        :return: A Pandas DataFrame containing the extracted data.
        """
        try:
            # Use the Pandas read_sql function to extract data from the 'customer_activity' table
            query = "SELECT * FROM customer_activity"
            df = pd.read_sql(query, self.engine)
            print("Data extracted successfully from 'customer_activity'.")
            return df
        except Exception as e:
            print(f"Error extracting data from 'customer_activity': {e}")
            raise

    def save_to_csv(self, dataframe, filename='customer_activity_data.csv'):
        """
        Save the Pandas DataFrame to a CSV file on the local machine.
        
        :param dataframe: The Pandas DataFrame to save.
        :param filename: The name of the file to save the data to (default is 'customer_activity_data.csv').
        """
        try:
            # Save the DataFrame to a CSV file in the current working directory
            dataframe.to_csv(filename, index=False)
            print(f"Data successfully saved to {filename}.")
        
        except Exception as e:
            print(f"Error saving data to CSV: {e}")
            raise

    def load_local_data(filename='customer_activity_data.csv'):
        """
        Load data from a locally stored CSV file into a Pandas DataFrame.
        
        :param filename: The name of the CSV file to load (default is 'customer_activity_data.csv').
        :return: A Pandas DataFrame containing the loaded data.
        """
        try:
            # Load the CSV file into a Pandas DataFrame
            dataframe = pd.read_csv(filename)
            
            # Print the shape of the data
            print(f"Data shape: {dataframe.shape}")
            
            # Print a preview of the data
            print("Data sample:")
            print(dataframe.head())
            
            return dataframe
        except Exception as e:
            print(f"Error loading data from {filename}: {e}")
            raise
