import os
import pandas as pd
import json
import logging

def load_data(file_path: str, file_type: str = 'csv') -> pd.DataFrame:
    """
    Loading data from file.
    :param file_path: Path to the file
    :param file_type: Type of the file ('csv', 'json', 'excel')
    :return: DataFrame containing the loaded data
    """
    if file_type == 'csv':
        data = pd.read_csv(file_path)
    elif file_type == 'json':
        data = pd.read_json(file_path)
    elif file_type == 'excel':
        data = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    logging.info(f"Data loaded from {file_path}")
    return data

def save_data(data: pd.DataFrame, file_path: str, file_type: str = 'csv'):
    """
    Saving data to file.
    :param data: DataFrame containing the data to save
    :param file_path: Path to the file
    :param file_type: Type of the file ('csv', 'json', 'excel')
    """
    if file_type == 'csv':
        data.to_csv(file_path, index=False)
    elif file_type == 'json':
        data.to_json(file_path, orient='records', lines=True)
    elif file_type == 'excel':
        data.to_excel(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    logging.info(f"Data saved to {file_path}")

def save_to_file(data, filename: str):
    """
    Saving data to file (JSON/plain text).
    :param data: Data to save
    :param filename: Name of the file
    """
    with open(filename, 'w') as file:
        if isinstance(data, (dict, list)):
            json.dump(data, file, indent=4)
        else:
            file.write(str(data))
    logging.info(f"Data saved to {filename}")

def read_from_file(filename: str):
    """
    Reading data from file (JSON/plain text).
    :param filename: Name of the file
    :return: Data read from the file
    """
    with open(filename, 'r') as file:
        if filename.endswith('.json'):
            return json.load(file)
        else:
            return file.read()

def create_directory(path: str):
    """
    Creating directory if it does not exist.
    :param path: Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    logging.info(f"Directory created at {path}")

if __name__ == "__main__":
    create_directory('example_dir')

    sample_data = {
        'feature1': [1, 2, 3],
        'feature2': ['A', 'B', 'C'],
        'label': [0, 1, 0]
    }
    df = pd.DataFrame(sample_data)
    
    save_data(df, 'example_dir/sample_data.csv', file_type='csv')
    loaded_df = load_data('example_dir/sample_data.csv', file_type='csv')
    print("Loaded DataFrame:\n", loaded_df)
    
    sample_dict = {'name': 'Deepfake Detection', 'version': '1.0'}
    save_to_file(sample_dict, 'example_dir/sample_data.json')
    loaded_dict = read_from_file('example_dir/sample_data.json')
    print("Loaded JSON:\n", loaded_dict)
