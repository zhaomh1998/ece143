# Library imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.io as pio
import plotly.offline as py
import plotly.graph_objs as go

import preprocessing

py.init_notebook_mode(connected=True)
pd.options.mode.chained_assignment = None

# Matplotlib plotting setting
SMALL_SIZE = 15
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


class Plot:
    """Plot class"""

    def __init__(self):
        '''Get datasets from preprocessing'''
        self.training = preprocessing.KaggleSet('train').get_dataset()
        self.test = preprocessing.KaggleSet('test').get_dataset()
        self.hh_char = preprocessing.INECDataSet('household_characteristics').get_dataset()
        self.education = preprocessing.INECDataSet('edu').get_dataset()
        self.edu_level = preprocessing.INECDataSet('edu_lvl').get_dataset()
        # Verify successful load of datasets
        assert not self.training.empty
        assert not self.test.empty
        assert not self.hh_char.empty
        assert not self.education.empty
        assert not self.edu_level.empty

        training_grps = self.training.groupby('Target')
        self.training_grp = {1: training_grps.get_group(1),
                             2: training_grps.get_group(2),
                             3: training_grps.get_group(3),
                             4: training_grps.get_group(4)}

    def plot_avg_num_per_hh(self):
        '''PLots average number of children / adult per household, using plotly'''
        avg_child = [ig['hogar_nin'].sum() / ig.shape[0] for ig in self.training_grp.values()]
        avg_adult = [ig['hogar_adul'].sum() / ig.shape[0] for ig in self.training_grp.values()]
        children = go.Bar(
            x=[1, 2, 3, 4],
            y=avg_child,
            name='Children'
        )
        adult = go.Bar(
            x=[1, 2, 3, 4],
            y=avg_adult,
            name='Adults'
        )

        layout = go.Layout(
            xaxis={'title': 'Income Group'},
            yaxis={'title': 'Average number per household'},
            barmode='stack'
        )

        fig = go.Figure(data=[children, adult], layout=layout)  # , layout=layout)
        py.iplot(fig)

    def plot_avg_income_change_since_2010(self):
        '''Plots average income per household's %change since 2010'''
        average_income_household = Plot.get_region(self.hh_char, 'Category',
                                                   ['Income per household', 'Average income per household',
                                                    'Average total household income'])
        average_income_household_person = Plot.get_region(self.hh_char, 'Category', ['Per capita income per household',
                                                                                     'Average household income per capita'])
        average_income_household.loc[:,
        ['Total', 'Quintil 1', 'Quintil 2', 'Quintil 3', 'Quintil 4', 'Quintil 5']] = average_income_household.loc[:,
                                                                                      ['Total', 'Quintil 1',
                                                                                       'Quintil 2', 'Quintil 3',
                                                                                       'Quintil 4',
                                                                                       'Quintil 5']].astype(float)
        average_income_household_person.loc[:, ['Total', 'Quintil 1', 'Quintil 2', 'Quintil 3', 'Quintil 4',
                                                'Quintil 5']] = average_income_household_person.loc[:,
                                                                ['Total', 'Quintil 1', 'Quintil 2', 'Quintil 3',
                                                                 'Quintil 4', 'Quintil 5']].astype(float)

        cr_avg_income_pct_change = Plot.get_region(average_income_household, 'region', ['Whole Costa Rica']).set_index(
            'year').drop(['Category', 'region'], axis=1)
        for year in range(2011, 2018):
            cr_avg_income_pct_change.loc[year, :] = (cr_avg_income_pct_change.loc[year,
                                                     :] / cr_avg_income_pct_change.loc[2010, :] - 1) * 100

        cr_avg_income_pct_change = cr_avg_income_pct_change.drop(2010)
        fig, ax = plt.subplots(figsize=(10, 7))
        cr_avg_income_pct_change['Quintil 1'].plot.line(ax=ax, lw=6)
        cr_avg_income_pct_change['Quintil 5'].plot.line(ax=ax, lw=6)
        ax.set_title('Average Income per household % change since 2010', fontsize=18)
        ax.legend(labels=['Quintil 1', 'Quintil 5'])
        ax.set_ylabel('percent change');
        plt.savefig(os.getcwd() + '/plots/Income percent change quintil.png')

    def plot_avg_hh_income_pc_b(self):
        '''Plots average household income in region Pacifico Central and Brunca'''

        # Get the same columns (looks different because of translation)
        float_cols = ['Total', 'Quintil 1', 'Quintil 2', 'Quintil 3', 'Quintil 4', 'Quintil 5']
        average_income_household = Plot.get_region(self.hh_char, 'Category',
                                                   ['Income per household', 'Average income per household',
                                                    'Average total household income'])

        average_income_household[float_cols] = average_income_household[float_cols].astype(float)

        c_average_income_household = average_income_household.copy()

        c_average_income_household = c_average_income_household.loc[
                                     (c_average_income_household['region'] == 'Pacífico Central') |
                                     ((c_average_income_household['region'] == 'Brunca')), :]

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.lineplot(x="year", y="Total", hue="region", data=c_average_income_household, ax=ax, lw=6)
        ax.set_title('Average Household Income', fontsize=20)
        plt.savefig(os.getcwd() + '/plots/Avg_Income_Parallel_1.png');

    def plot_avg_hh_income_p_pc_b(self):
        '''PLots average household per person income in region Pacifico Central and Brunca'''
        float_cols = ['Total', 'Quintil 1', 'Quintil 2', 'Quintil 3', 'Quintil 4', 'Quintil 5']
        average_income_household_person = Plot.get_region(self.hh_char, 'Category', ['Per capita income per household',
                                                                                     'Average household income per capita'])
        average_income_household_person[float_cols] = average_income_household_person[float_cols].astype(float)
        c_average_income_household_person = average_income_household_person.copy()
        c_average_income_household_person = c_average_income_household_person.loc[
                                            (c_average_income_household_person['region'] == 'Pacífico Central') |
                                            ((c_average_income_household_person['region'] == 'Brunca')), :]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.lineplot(x="year", y="Total", hue="region", data=c_average_income_household_person, ax=ax, lw=6)
        ax.set_title('Average Household Income per Person', fontsize=20)
        plt.savefig(os.getcwd() + '/plots/Avg_Income_Parallel_2.png')

    def plot_overcrowding(self):
        '''Plots person per room for household in 4 poverty levels'''
        fig, ax = plt.subplots(figsize=[10, 7])
        sns.boxplot(x='Target', y='overcrowding', data=self.training, ax=ax)
        ax.set_ylabel('Person per Room')
        ax.set_xlabel('Poverty Level')
        ax.set_title('Person per Room for Households in Four Poverty Levels', fontsize=18)
        plt.savefig(os.getcwd() + '/plots/household_overcrowding.png')

    def plot_poverty_laura(self):
        '''Plots the total poverty % in whole costa rica under president Laura'''
        year = list(range(2010, 2018))
        pov = [21.2, 21.7, 20.6, 20.7, 22.4, 21.7, 20.5, 20.0]
        poverty_old = go.Scatter(
            x=year[:5],
            y=pov[:5],
            mode='lines+markers',
            name='% Poverty',
            line=dict(width=4)
        )
        layout = go.Layout(
            width=500,
            height=500,
            xaxis=dict(title='Year'),
            yaxis=dict(title='Total poverty %'),
            title='Costa Rica Total Poverty % under Laura Chinchilla',
            titlefont=dict(size=14))

        fig = go.Figure(data=[poverty_old], layout=layout)
        py.iplot(fig)

    def plot_poverty_luis(self):
        '''Plots the total poverty % in the whole costa rica under president luis'''
        year = list(range(2010, 2018))
        pov = [21.2, 21.7, 20.6, 20.7, 22.4, 21.7, 20.5, 20.0]
        poverty_new = go.Scatter(
            x=year[4:],
            y=pov[4:],
            mode='lines+markers',
            name='% Poverty',
            line=dict(
                color=('green'),
                width=4)
        )
        layout = go.Layout(
            width=500,
            height=500,
            xaxis=dict(title='Year'),
            yaxis=dict(title='Total poverty %'),
            title='Costa Rica Total Poverty % under Luis Guillermo Solis',
            titlefont=dict(size=14))

        fig = go.Figure(data=[poverty_new], layout=layout)
        py.iplot(fig)

    def plot_edu_level(self):
        self.training['instlevel'] = self.training['instlevel1'] * 1 + self.training['instlevel2'] * 2 + self.training[
            'instlevel3'] * 3 + \
                                     self.training['instlevel4'] * 4 + self.training['instlevel5'] * 5 + self.training[
                                         'instlevel6'] * 6 + \
                                     self.training['instlevel7'] * 7 + self.training['instlevel8'] * 8 + self.training[
                                         'instlevel9'] * 9
        inst = pd.DataFrame()
        for target in range(1, 5):
            inst[target] = self.training.loc[self.training['Target'] == target, 'instlevel'].value_counts().sort_index()
            total_samples = inst[target].sum()
            inst[target] = inst[target].apply(lambda x: x / total_samples)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title('Distribution of Education Level in Four Income Groups', fontsize=18)
        # ax.set_xticks(['a','a','a','a','a','a','a','a'],x)
        x_ticks_labels = ['No Edu', 'Primary WIP', 'Primary',
                          'Secondary WIP', 'Secondary',
                          'Technical Secondary WIP', 'Technical Secondary',
                          'Undergrad/Higher Ed', 'Postgrad']
        # ax.set_xticks(x_ticks_labels)
        ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=12)
        inst.plot.line(ax=ax);
        plt.xticks(rotation=20);
        ax.legend(title='Income Level');
        plt.savefig(os.getcwd() + '/plots/edu.png')

    def plot_economy_income(self):
        # hh_char[hh_char.Category == 'Income per household'][hh_char.region == 'Whole Costa Rica']
        float_cols = ['Total', 'Quintil 1', 'Quintil 2', 'Quintil 3', 'Quintil 4', 'Quintil 5']
        average_income_household = Plot.get_region(self.hh_char, 'Category',
                                                   ['Income per household', 'Average income per household',
                                                    'Average total household income'])
        average_income_household.loc[:, float_cols] = average_income_household[float_cols].astype(float)
        central_income = average_income_household.loc[
            (average_income_household.region == 'Central') | (average_income_household.region == 'Pacífico Central')]
        #
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.lineplot(x='year', y='Total', hue='region', data=central_income, lw=6)
        ax.set_title('Average Household Income', fontsize=20)
        ax.set_ylabel('Income')
        plt.savefig(os.getcwd() + '/plots/economy_income.png')

    @staticmethod
    def get_region(df, region_col_name, regions_to_get):
        """
        Given a pandas DataFrame, get the <regions_to_get> from <region_col_name>. Return the result as a DataFrame
        :param df: Pandas DataFrame to work on
        :type df: Pandas DataFrame
        :param region_col_name: Name of the column to get region from
        :type region_col_name: str
        :param regions_to_get: List of regions to get from df
        :type regions_to_get: List of str
        """
        assert isinstance(df, pd.core.frame.DataFrame)
        assert not df.empty
        assert isinstance(region_col_name, str)
        assert region_col_name in df.columns, f'{region_col_name} not in dataframe columns'
        region_grp = df.groupby(region_col_name)
        assert [region in region_grp.groups.keys() for region in
                regions_to_get], '{regions_to_get} not found in column {region_col_name}!'

        region_df = pd.concat([region_grp.get_group(region_name) for region_name in regions_to_get])
        assert not region_df.empty
        return region_df
