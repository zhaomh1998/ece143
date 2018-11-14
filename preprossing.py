import pandas as pd


def load_csv(data_path):
    """
    Loads the data set and return a pandas DataFrame
    :return: pandas DataFrame
    """
    from os import path
    assert isinstance(data_path, str)
    assert path.exists(data_path), f'{data_path} does not exist!'
    assert data_path[-4:] == '.csv', 'Input file has to be csv!'
    return pd.read_csv(data_path)


if __name__ == '__main__':
    data = load_csv('train.csv')
    print(data)