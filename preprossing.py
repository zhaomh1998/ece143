import pandas as pd


def load_csv(data_path):
    """
    Loads the data set and return a pandas DataFrame
    :param str data_path: Path to data csv file
    :return: pandas DataFrame read from the file
    """
    from os import path
    assert isinstance(data_path, str)
    assert path.exists(data_path), f'{data_path} does not exist!'
    assert data_path[-4:] == '.csv', 'Input file has to be csv!'
    return pd.read_csv(data_path)


def clean_data(original_data):
    """
    Filter out unwanted and invalid data from the data set
    :param DataFrame original_data: Original DataFrame read from data csv
    :return: Filtered pandas DataFrame
    """
    assert isinstance(original_data, pd.DataFrame)
    filtered_data = original_data.copy()
    filtered_data.drop(['v18q1'], axis=1, inplace=True)
    # TODO: Add other filtering
    return filtered_data


def get_training_set():
    """
    Load training.csv and clean it
    :return: Cleaned train.csv pandas DataFrame
    """
    return clean_data(load_csv('train.csv'))


if __name__ == '__main__':
    data = get_training_set()
    print(data)
