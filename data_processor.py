import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def __init__(self):
        self.df_train = None
        self.df_test = None
        self.df_store = None
        self.scale_y = None

    '''importing data'''

    def load_data(self, path):
        self.df_train = pd.read_csv(path+'/train.csv', parse_dates=['Date'], dtype={'StateHoliday': 'category'})
        self.df_test = pd.read_csv(path+'/test.csv', parse_dates=['Date'], dtype={'StateHoliday': 'category'})
        self.df_store = pd.read_csv(path+'/store.csv')

    '''sorting data to prepare a timeseries'''
    def preprocessing(self, n_input):
        self.df_train.sort_values('Date', ascending=True, inplace=True)
        self.df_test.sort_values('Date', ascending=True, inplace=True)

        '''joining train and test data to manipulate them together'''
        train_test = pd.concat([self.df_train, self.df_test])

        '''to keep the shape same between train and test, filling in na for all test rows'''
        train_test.loc[train_test['Sales'].isna(), 'Sales'] = -1

        '''Splitting date into day, month, year and weekofyear'''
        train_test['Month'] = train_test['Date'].dt.month
        train_test['Year'] = train_test['Date'].dt.year
        train_test['Day'] = train_test['Date'].dt.day
        train_test['WeekOfYear'] = train_test['Date'].dt.weekofyear

        '''
        # df_open = self.df_train.loc[df_train['Open'] == 0]
        # df_open.groupby('DayOfWeek')['Open'].describe()
        # df_open = self.df_test.loc[df_test['Open'] == 0]
        # df_open.groupby('DayOfWeek')['Open'].describe()
        In test set, values are missing for Column = Open, based on the trend uing train set, it's concluded that shops 
        remain open on week days mostly, hence filling these missing values with 1
        '''

        train_test['Open'].fillna(1, inplace=True)

        '''store file has 3 missing values for CompetitionDistance, filling in these with the median'''
        self.df_store['CompetitionDistance'].fillna(self.df_store['CompetitionDistance'].median(), inplace=True)

        '''merging store data with train and test concatenated dataset'''
        train_test_merged = pd.merge(train_test, self.df_store, on='Store', how='left')

        '''Evaluating CompetitionOpenMonths and PromoOpenMonths'''
        train_test_merged['CompetitionOpenMonths'] = 12 * (
                train_test_merged['Year'] - train_test_merged['CompetitionOpenSinceYear']) + train_test_merged[
                                                         'Month'] - \
                                                     train_test_merged['CompetitionOpenSinceMonth']
        train_test_merged['PromoOpenMonths'] = 12 * (
                train_test_merged['Year'] - train_test_merged['Promo2SinceYear']) + (
                                                       train_test_merged['WeekOfYear'] - train_test_merged[
                                                   'Promo2SinceWeek']) / 4.0

        train_test_merged['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
        train_test_merged['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
        train_test_merged['CompetitionOpenSinceYear'].fillna(0, inplace=True)
        train_test_merged['Promo2SinceWeek'].fillna(0, inplace=True)
        train_test_merged['Promo2SinceYear'].fillna(0, inplace=True)
        train_test_merged['PromoInterval'].fillna(0, inplace=True)
        train_test_merged['CompetitionOpenMonths'].fillna(0, inplace=True)
        train_test_merged['PromoOpenMonths'].fillna(0, inplace=True)

        '''Splitting train and test for separate evaluation and processing'''
        train_data = train_test_merged.loc[:self.df_train.index.size - 1, :]
        test_data = train_test_merged.loc[self.df_train.index.size:, :]

        '''
        #train_data[train_data['Customers'] != 0].groupby(['StoreType', 'DayOfWeek'])['Sales', 'Customers'].sum()
        
        Based on the above result, finding 
        1. average Sales per storetype per dayofweek
        2. average number of customers per storetype per dayofweek
        '''
        df_avg = pd.DataFrame(train_data[train_data['Customers'] != 0].groupby(['StoreType', 'DayOfWeek']).apply(
            lambda x: x['Sales'].sum() / x['Customers'].sum()))
        df_avg_cust = pd.DataFrame(
            train_data[train_data['Customers'] != 0].groupby(['StoreType', 'DayOfWeek'])['Customers'].mean())
        df_avg_cust.columns = ['AvgCustomer']
        df_avg.columns = ['AvgSalesPCustomer']
        train_data = train_data.merge(df_avg, on=['StoreType', 'DayOfWeek'], how='left')
        train_data = train_data.merge(df_avg_cust, on=['StoreType', 'DayOfWeek'], how='left')
        test_data = test_data.merge(df_avg, on=['StoreType', 'DayOfWeek'], how='left')
        test_data = test_data.merge(df_avg_cust, on=['StoreType', 'DayOfWeek'], how='left')

        '''Filling Na'''
        test_data['AvgCustomer'].fillna(0, inplace=True)
        test_data['AvgSalesPCustomer'].fillna(0, inplace=True)
        train_data['AvgCustomer'].fillna(0, inplace=True)
        train_data['AvgSalesPCustomer'].fillna(0, inplace=True)

        '''With the help of a key map for the months, finding out those months in which promo was active'''
        month2str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                     7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        train_data['monthStr'] = train_data.Month.map(month2str)
        test_data['monthStr'] = test_data.Month.map(month2str)

        train_data['IsPromoMonth'] = 0
        for interval in train_data.PromoInterval.unique():
            interval = str(interval)
            if interval != '':
                for month in interval.split(','):
                    train_data.loc[
                        (train_data.monthStr == month) & (train_data.PromoInterval == interval), 'IsPromoMonth'] = 1

        test_data['IsPromoMonth'] = 0
        for interval in test_data.PromoInterval.unique():
            interval = str(interval)
            if interval != '':
                for month in interval.split(','):
                    test_data.loc[
                        (test_data.monthStr == month) & (test_data.PromoInterval == interval), 'IsPromoMonth'] = 1

        '''Checking data types at this state to make sure everything in float for model to process
        #test_data.dtypes
        In case of StateHoliday one values is numeric and others are string. 
        In order to get dummies, changing numeric to string 
        '''

        train_data.loc[train_data['StateHoliday'] == 0, 'StateHoliday'] = 'd'
        test_data.loc[test_data['StateHoliday'] == 0, 'StateHoliday'] = 'd'

        train_data = pd.get_dummies(train_data, columns=["StateHoliday", "StoreType", "Assortment"], drop_first=False)
        test_data = pd.get_dummies(test_data, columns=["StateHoliday", "StoreType", "Assortment"], drop_first=False)

        '''Preparing a list of columns in order, to feed into the model'''
        cols_num = ["Sales", "DayOfWeek", "Open", "Promo", "SchoolHoliday", "CompetitionDistance",
                    "CompetitionOpenSinceMonth", "Promo2",
                    "Promo2SinceWeek", "AvgSalesPCustomer", "AvgCustomer", "Month", "Day",
                    "CompetitionOpenMonths", "PromoOpenMonths", "IsPromoMonth", "Store", 'StateHoliday_0',
                    'StateHoliday_a',
                    'StateHoliday_b', 'StateHoliday_c', 'StoreType_a', 'StoreType_b',
                    'StoreType_c', 'StoreType_d', 'Assortment_a', 'Assortment_b', 'Assortment_c']

        '''In case of test data there StateHoliday type b and c are missing, in order to keep the shape same
        between train and test set, adding null columns'''
        test_data['StateHoliday_b'] = 0
        test_data['StateHoliday_c'] = 0

        '''Forming desired data sets for train and test with all the necessary columns in desired order'''
        train_data1 = train_data[cols_num]
        test_data1 = test_data[cols_num]

        '''Adding data worth n_input size from train to test to get prediction for the time series'''
        test_data1 = pd.concat([train_data1.iloc[-n_input:, :], test_data1])

        '''Applying min max to normalize the data.
        Keeping different fitness function for train features, train label and test features
        '''

        scale_x = MinMaxScaler()
        self.scale_y = MinMaxScaler()
        scale_test = MinMaxScaler()

        x_train = scale_x.fit_transform(train_data1.astype(float))
        y_train = self.scale_y.fit_transform(train_data['Sales'].astype(float).values.reshape(-1, 1))
        x_test = scale_test.fit_transform(test_data1.astype(float))

        '''Splitting train data into train and validation set'''
        split_idx = math.floor(len(x_train) * 0.8)

        x_val = x_train[split_idx:]
        y_val = y_train[split_idx:]
        x_train = x_train[:split_idx]
        y_train = y_train[:split_idx]
        return x_train, x_val, y_train, y_val, x_test

    def generate_result(self, predict, path):
        predict1 = self.scale_y.inverse_transform(predict)
        df_result = pd.DataFrame()
        df_result['Id'] = self.df_test['Id']
        df_result['Sales'] = predict1.reshape(-1, )
        df_result.sort_values('Id', ascending=True, inplace=True)
        df_result.to_csv(path+'/result.csv', index=False)
