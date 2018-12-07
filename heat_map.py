import os
import pandas as pd
import gmaps

COMPANIES_DATASET_PATHS = [os.getcwd() + '/datasets/companies_' + str(year) + '.xls' for year in range(2012, 2018)]


def get_city_location():
    """
    Process and returns the location of cities in Costa Rica from geonames database
    https://www.geonames.org/export/
    :return: Pandas DataFrame
    """
    city_locations = pd.read_csv(os.getcwd() + '/datasets/CR.txt', sep='\t', header=None)
    city_locations = city_locations.loc[:, 1:5].drop([2, 3], axis=1).rename({1: 'name', 4: 'lat', 5: 'long'}, axis=1)
    city_locations = city_locations.drop_duplicates(subset='name')
    city_locations['location'] = list(city_locations[['lat', 'long']].itertuples(index=False, name=None))
    city_locations = city_locations.drop(['lat', 'long'], axis=1)

    location_table = pd.Series(city_locations['location'].tolist(), index=city_locations['name'])
    return location_table


def extract_province_indexes(ser, start=None, end=None):
    """
    Find pandas Series Indexes to extract data from a certain range starting (ending) with format NaN, province, Nan
    :param ser: Pandas Series to work on
    :type ser: Pandas Series
    :param start: name to look for for start
    :type start: same as ser.dtype
    :param end: name to look for for end
    :type end: same as ser.dtype
    :return: tuple of Pandas Series Index for where the data is located, as (start, end)
    """
    assert isinstance(ser, pd.core.series.Series), 'Input a Pandas Series'
    assert any([start, end]), 'Must specify at least one of start / end'

    if all([start, end]):
        start_index = locate_province(ser, start, 'end')
        end_index = locate_province(ser, end, 'start')
    elif start == None:
        start_index = [ser.index[0]]
        end_index = locate_province(ser, end, start)
    elif end == None:
        start_index = locate_province(ser, start, 'end')
        end_index = [ser.index[-1]]
    assert len(start_index) == 1, f'Could not find {start}'
    assert len(end_index) == 1, f'Could not find {end}'
    assert start_index[0] < end_index[0], f'{start}\'s index <= {end}\'s index!'
    return (start_index[0], end_index[0])


def locate_province(ser, province_name, index_type):
    """
    Find the index where pattern NaN, province_name, NaN appears, and return the index in tuple.
    :param ser: Pandas Series to work on
    :type ser: Pandas Series
    :param province_name: Name of the province
    :type province_name: str to get the beginning or end index for the pattern, choose from 'start' or 'end'
    :param index_type: str
    """
    assert isinstance(ser, pd.core.series.Series), 'Input a Pandas Series'
    assert isinstance(province_name, str)
    assert isinstance(index_type, str)
    assert index_type in ['start', 'end']
    if index_type == 'end':
        return ser.loc[(ser.shift(2).isnull()) &
                       (ser.shift(1) == province_name) &
                       (ser.isnull())].index
    elif index_type == 'start':
        return ser.loc[(ser.isnull()) &
                       (ser.shift(-1) == province_name) &
                       (ser.shift(-2).isnull())].index


def gmap_heat_map(loc_list, info):
    """
    Take in a list of tuples (latitude, longitude) heat map using gmap.
    gmap's API key need to be stored in api_keys.py in the same folder. Just put in a variable gmap_key = 'AI...'
    :param loc_list: List of tuples (latitude, longitude) to be plotted
    :type loc_list: list of tuples
    :param info: The string to be shown on the plot
    :type info: str
    """
    import api_keys
    assert len(loc_list) > 0, 'Location list cannot be empty!'
    gmaps.configure(api_key=api_keys.gmap_key)
    fig = gmaps.figure(map_type='HYBRID')
    fig.add_layer(gmaps.heatmap_layer(loc_list, opacity=0.8, point_radius=20))
    fig.add_layer(
        gmaps.symbol_layer([(10.8, -83.0)], fill_color='red', stroke_color='red', info_box_content=info, scale=1))
    return fig


def load_companies_data():
    """
    Load the companies data and return processed Pandas DataFrame
    :return: Dictionary containing all city's location data and number of companies, ready to make hear map
    """
    year_dict = dict()
    costa_rica_provinces = ['SAN JOSÉ', 'ALAJUELA', 'CARTAGO', 'HEREDIA', 'GUANACASTE', 'PUNTARENAS', 'LIMÓN']
    for year_df, year in zip(COMPANIES_DATASET_PATHS, [year for year in range(2012, 2018)]):
        cp = pd.read_excel(year_df).drop('Unnamed: 0', axis=1)
        cp = cp.rename(dict(zip(list(cp.columns), ['region', 'value'])), axis=1)
        cp.loc[:, 'region'] = cp['region'].str.strip()
        cp['year'] = year
        year_dict[year] = cp

    for year, df in year_dict.items():
        # Get rid of all province, canton totals, leave all the cities
        all_cities = pd.DataFrame()
        # All cantons except last
        for city_start, city_end in zip(costa_rica_provinces[:-1], costa_rica_provinces[1:]):
            (index_start, index_end) = extract_province_indexes(df['region'], start=city_start, end=city_end)
            prov = df[index_start:index_end]
            cantons_index = prov[prov.isnull()['region']].index + 1
            cities = prov.drop(cantons_index, axis=0).dropna().reset_index(drop=True)
            all_cities = all_cities.append(cities)
        # Last canton
        (index_start, index_end) = extract_province_indexes(df['region'], start=costa_rica_provinces[-1])
        prov = df[index_start:index_end]
        cantons_index = prov[prov.isnull()['region']].index + 1
        cantons_index = cantons_index if cantons_index[-1] <= prov.index[-1] else cantons_index[:-1]
        cities = prov.drop(cantons_index, axis=0).dropna().reset_index(drop=True)
        all_cities = all_cities.append(cities)

        year_dict[year] = all_cities  # Overwrite year_dict with the processed DataFrame

    # Finds each city's coordinate
    location_table = get_city_location()
    years_locations = dict()
    for year, cp in year_dict.items():
        locations = []
        for index, n_companies in cp.reset_index(drop=True).iterrows():
            if n_companies['region'] in location_table.keys():
                locations.append([location_table[n_companies['region']]] * n_companies['value'])
        years_locations[year] = [item for sublist in locations for item in sublist]  # Flatten location array
    return years_locations
