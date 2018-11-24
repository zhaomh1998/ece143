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

def read_var(fname):
    '''This function reads the variable description from the given text file for our dataset. The text in the file 
    should have the first word as the variable name and the following words as the description of the variable.
    input:
    fname-->file name
    output:
    variable description as a dictionary
    '''
    feature={}
    assert isinstance(fname,str),"The given file name is not a string"
    with open(fname,"r") as file:
        for data in file.read().splitlines():
            variable=data.split()[0].replace(",","")
            feature[variable]=' '.join(data.split()[1:])
    return feature

def data_descrip(data):
    '''
    This function gives the column descriptions of the training data given
    input:
    data--> data as panda dataframe
    output:
    data key description
    '''
    assert isinstance(data, pandas.core.series.Series),"the data is not in pandas dataframe format"
    feature = read_var("variable.txt")
    return feature[data.name]
    
    
def get_training_set():
    """
    Load training.csv and clean it
    :return: Cleaned train.csv pandas DataFrame
    """
    return clean_data(load_csv('train.csv'))


if __name__ == '__main__':
    data = get_training_set()
    renamed_data = rename_data(data)
    print(data)
