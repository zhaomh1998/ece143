import os
import pandas as pd
from Data_scraping import *

# INEC Data set constants
DATA_SET_FOLDER = os.getcwd() + '/datasets/'
DATA_SET_PATHS = {'household_characteristics': DATA_SET_FOLDER + 'household_characteristics_2010_2017.xls',
                  'poverty_level_stat': DATA_SET_FOLDER + 'poverty_level_by_zone_2010_2017.xls',
                  'edu': [DATA_SET_FOLDER + 'edu_' + str(year) + '.xlsx' for year in range(2010, 2018)],
                  'edu_lvl': [DATA_SET_FOLDER + 'edu_lvl_' + str(year) + '.xlsx' for year in range(2010, 2018)]}

HOUSEHOLD_CHAR_COLS = {'Cuadro 2': 'Category', 'Unnamed: 2': 'Total', 'Unnamed: 3': 'Quintil 1',
                       'Unnamed: 4': 'Quintil 2', 'Unnamed: 5': 'Quintil 3', 'Unnamed: 6': 'Quintil 4',
                       'Unnamed: 7': 'Quintil 5'}

EDU_COLS = ['region', '5-24 total', '5-24 not attend%', '5-24 attend%',
            '5-12 total', '5-12 not attend%', '5-12 attend%',
            '13-17 total', '13-17 not attend%', '13-17 attend%',
            '18-24 total', '18-24 not attend%', '18-24 attend%']

EDU_LVL_COLS = ['region', 'total', 'no education', 'incomplete primary', 'complete primary',
                'secondary incomplete', 'secondary complete', 'technical secondary incomplete',
                'technical secondary complete', 'undergraduate', 'postgraduate', 'no response']

REGIONS = ['Central', 'Chorotega', 'Pacífico Central', 'Brunca', 'Huetar Atlántica', 'Huetar Norte']


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


def read_var(fname='variable.txt'):
    """This function reads the variable description from the given text file for our dataset. The text in the file
    should have the first word as the variable name and the following words as the description of the variable.
    input:
    fname-->file name
    output:
    variable description as a dictionary
    """
    feature = {}
    assert isinstance(fname, str), "The given file name is not a string"
    with open(fname, "r") as file:
        for data in file.read().splitlines():
            variable = data.split()[0].replace(",", "")
            feature[variable] = ' '.join(data.split()[1:])
    return feature


def data_descrip(data):
    """
    This function gives the column descriptions of the training data given
    input:
    data--> data as panda DataFrame
    output:
    data key description
    """
    assert isinstance(data, pd.core.series.Series), "the data is not in pandas dataframe format"
    feature = read_var("variable.txt")
    return feature[data.name]


def get_training_set():
    """
    Load training.csv and clean it
    :return: Cleaned train.csv pandas DataFrame
    """
    return clean_data(load_csv(os.getcwd() + '/datasets/train.csv'))


def get_test_set():
    """
    Load test.csv and clean it
    :return: Cleaned train.csv pandas DataFrame
    """
    return clean_data(load_csv(os.getcwd() + '/datasets/test.csv'))


class INECDataSet:
    """Class for all http://inec.go.cr related data sets processing"""

    def __init__(self, dataset):
        """
        :param dataset: Name of the data set
        :type dataset: str
        :return: Instance of INECDataSet
        """
        assert isinstance(dataset, str)
        assert dataset in DATA_SET_PATHS.keys(), 'Please choose from {DATA_SET_PATHS.keys()}'

        self.df = None
        self.description = None
        self.category_translation = dict()
        print('Start to load data...')
        if dataset == 'household_characteristics':
            self.process_household_characteristics(DATA_SET_PATHS['household_characteristics'])
        elif dataset == 'poverty_level_stat':
            self.process_poverty_level(DATA_SET_PATHS['poverty_level_stat'])
        elif dataset == 'edu':
            self.process_edu(DATA_SET_PATHS['edu'])
        elif dataset == 'edu_lvl':
            self.process_edu_lvl(DATA_SET_PATHS['edu_lvl'])

    def process_household_characteristics(self, dataset_path):
        """
        Load and process the household characteristics data set, and save in self.df
        :param dataset_path: Path to the data set
        :type dataset_path: str
        """
        assert isinstance(dataset_path, str)
        assert os.path.exists(dataset_path), '{dataset_path} does not exist!'
        self.df = pd.read_excel(dataset_path).rename(HOUSEHOLD_CHAR_COLS, axis=1)  # Read xls data set and rename cols
        self.description = string(self.df['Category'][1:7].str.cat(sep='\n')).translate()
        # Drop Detaset description
        self.df = self.df.drop('Unnamed: 0', axis=1).dropna(subset=['Category'])[6:]
        # Clean inconsistent region naming
        self.df.loc[self.df.Category == 'Huetar Caribe', 'Category'] = 'Huetar Atlántica'
        # Separate years
        year_dict = self.slice_years_to_dict(self.df, 'Category', 2010, 2017)
        print('Processing... Please wait...')
        # Process every year and attach to final DataFrame
        self.df = pd.DataFrame()
        for year, year_df in year_dict.items():
            year_df = self.process_region(year_df, REGIONS)
            year_df = year_df.dropna()
            for entry in year_df['Category']:
                if entry not in self.category_translation.keys():
                    self.category_translation[entry] = string(
                        ''.join([i for i in str(entry) if
                                 not (i.isdigit() or i == '/')])).translate()  # Get rid of numbers and /
            year_df.loc[:, 'Category'] = year_df.loc[:, 'Category'].apply(lambda x: self.category_translation[x])
            year_df.loc[:, 'year'] = year
            self.df = self.df.append(year_df)
        self.df = self.df.reset_index(drop=True)

    def process_poverty_level(self, dataset_path):
        """
        Load and process the poverty level data set, and save in self.df
        :param dataset_path: Path to the data set
        :param dataset_path: str
        """
        assert isinstance(dataset_path, str)
        assert os.path.exists(dataset_path), '{dataset_path} does not exist!'
        raise NotImplementedError()

    def process_edu(self, dataset_path_list):
        """
        Load and process the list of education data set, and save in self.df
        :param dataset_path_list: List of path to every education data set
        :type dataset_path_list: list of str
        """
        year_dict = dict()
        assert all([os.path.exists(dataset_path) for dataset_path in dataset_path_list])
        for file_path, year in zip(dataset_path_list[:-2], range(2010, 2016)):
            year_dict[year] = pd.read_excel(file_path)
        year_dict[2016] = pd.read_excel(pd.ExcelFile(dataset_path_list[-2]), 0)  # Read in corresponding table
        year_dict[2017] = pd.read_excel(pd.ExcelFile(dataset_path_list[-1]), 'Cuadro 5')
        # Obtain data set description
        self.description = string(year_dict[2015].iloc[:3, 1].str.cat(sep='\n')).translate()
        # Append each of the processed year DataFrame
        self.df = pd.DataFrame()
        for year, year_df in year_dict.items():
            year_df = year_df.drop('Unnamed: 0', axis=1).dropna()
            year_df = year_df.rename(
                dict(zip(list(year_df.columns), EDU_COLS)), axis=1)
            year_df.loc[:, 'year'] = year
            self.df = self.df.append(year_df)
        self.df = self.df.reset_index(drop=True)
        self.df['region'] = self.df['region'].apply(lambda x: x.lstrip())  # Kill extra space in region
        self.df.loc[self.df.region == 'Huetar Caribe', 'region'] = 'Huetar Atlántica'  # Correct old region name

    def process_edu_lvl(self, dataset_path_list):
        """
        Load and process the list of education level data set, and save in self.df
        :param dataset_path_list: List of path to every education level data set
        :type dataset_path: list of str
        """
        year_dict = dict()
        assert all([os.path.exists(dataset_path) for dataset_path in dataset_path_list])
        for file_path, year in zip(dataset_path_list[:-2], range(2010, 2016)):
            year_dict[year] = pd.read_excel(file_path)
        year_dict[2016] = pd.read_excel(pd.ExcelFile(dataset_path_list[-2]), 1)  # Read in corresponding table
        year_dict[2017] = pd.read_excel(pd.ExcelFile(dataset_path_list[-1]), 'Cuadro 1')
        # Obtain data set description
        self.description = string(year_dict[2015].iloc[:3, 1].str.cat(sep='\n')).translate()
        # Append each of the processed year DataFrame
        self.df = pd.DataFrame()
        for year, year_df in year_dict.items():
            year_df = year_df.drop('Unnamed: 0', axis=1).dropna()
            year_df = year_df.rename(
                dict(zip(list(year_df.columns), EDU_LVL_COLS)), axis=1)
            # Add informational columns
            year_df['year'] = year
            year_df['male'] = (year_df['region'] == 'Hombres').astype(int)
            year_df['total'] = ((year_df['region'] != 'Hombres') & (year_df['region'] != 'Mujeres')).astype(int)
            for index, item in year_df['region'].iteritems():
                if item == 'Hombres':
                    year_df.loc[index, 'region'] = year_df['region'][index - 1]
                elif item == 'Mujeres':
                    year_df.loc[index, 'region'] = year_df['region'][index - 2]
            year_df = year_df.dropna()
            self.df = self.df.append(year_df)
        self.df = self.df.reset_index(drop=True)
        self.df.loc[self.df.region == 'Huetar Caribe', 'region'] = 'Huetar Atlántica'  # Correct old region name

    def get_dataset_description(self):
        """Returns data set description for the instance data set"""
        return self.description

    def get_dataset(self):
        """Returns DataFrame from this instance"""
        assert self.df is not None
        return self.df

    @staticmethod
    def slice_years_to_dict(df, column, year_start, year_end):
        """
        Slice df based on years into a dictionary
        :param df: DataFrame to operate on
        :type df: Pandas DataFrame
        :param column: Column name the year sits in
        :type column: Object
        :param year_start: Start year
        :type year_start: int
        :param year_end:  End year
        :type year_end: int
        :return: dict
        """
        assert not df.empty, 'DataFrame cannot be empty'
        assert column in df.columns, f'Column {column} not found in df!'
        assert isinstance(year_start, int)
        assert isinstance(year_end, int)
        assert year_start <= year_end

        year_dict = dict()
        end = 0
        # Split into every year and save every DataFrame into dict
        years_range = zip(range(year_start, year_end + 1), range(year_start + 1, year_end + 1))

        for (start, end), year in zip(
                [INECDataSet.extract_indexes(df['Category'], start_year, end_year) for (start_year, end_year) in
                 years_range],
                range(year_start, year_end)):
            year_dict[year] = df.loc[start + 1:end]

        year_dict[year_end] = df.loc[end + 1:]  # Add last year
        assert not year_dict[year_end].empty
        return year_dict

    @staticmethod
    def extract_indexes(ser, start, end):
        """
        Find pandas Series Indexes to extract data from a certain year in the format of start, ...data..., end
        :param ser: Pandas Series to work on
        :type ser: Pandas Series
        :param start: Object to locate for starting point
        :type start: same dtype as ser
        :param end: Object to locate for ending point
        :type end: same dtype as ser
        :return: List of Pandas Series Index for where the data is located
        """

        assert isinstance(ser, pd.core.series.Series), 'Input a Pandas Series'
        start_index = ser[ser == start]
        end_index = ser[ser == end]
        assert len(start_index) == 1, f'Could not find exactly one {start}'
        assert len(end_index) == 1, f'Could not find exactly one {end}'
        return (start_index.keys()[0], end_index.keys()[0])

    @staticmethod
    def process_region(year_df, regions):
        """
        Given the DataFrams year_df for the same year, kill the nan rows and add region to a new row
        :param year_df: Pandas Dataframe to work on
        :type year_df: Pandas DataFrame
        :param regions: List of regions string to slice
        :type regions: List of str
        :return: Processed Pandas DataFrame
        """
        assert isinstance(year_df, pd.core.frame.DataFrame)
        assert isinstance(regions, list)
        assert not year_df.empty
        assert len(regions) > 0
        for region in regions:
            region_result_count = len(year_df['Category'][year_df['Category'] == region].index)
            assert region_result_count == 1, f'{region_result_count} results found for region {region}!'
        # Process first region (Whole Costa Rica)
        start_index = year_df.index[0]
        end_index = year_df['Category'][year_df['Category'] == regions[0]].index[0]
        year_df.loc[year_df.loc[start_index:end_index].index, 'region'] = 'Whole Costa Rica'
        # Process the regions in the list
        for nth_region in range(len(regions) - 1):
            start_index = year_df['Category'][year_df['Category'] == regions[nth_region]].index[0]
            end_index = year_df['Category'][year_df['Category'] == regions[nth_region + 1]].index[0]
            year_df.loc[year_df.loc[start_index + 1:end_index].index, 'region'] = regions[nth_region]
        # Process the last region
        start_index = year_df['Category'][year_df['Category'] == regions[-1]].index[0]
        end_index = year_df.index[-1]
        year_df.loc[year_df.loc[start_index + 1:end_index + 1].index, 'region'] = regions[-1]
        return year_df
