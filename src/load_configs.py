"""
the load_configs.py module provides a function to load YAML configuration files.
It uses the PyYAML library to parse the YAML file and return its contents as a dictionary.
"""


from yaml import safe_load

def load_configs(file_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(file_path, 'r') as file:
        config = safe_load(file)
    return config