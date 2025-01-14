import yaml

def load_credentials(filepath="credentials.yaml"):
    """
    Load database credentials from a YAML file.
    
    :param filepath: Path to the credentials YAML file
    :return: A dictionary containing the database credentials
    """
    with open(filepath, 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials