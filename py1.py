# Python 1

        # print(tips.head(5))
   # total_bill   tip     sex smoker  day    time  size  recode
# 0       16.99  1.01  Female     No  Sun     NaN   2.0     0.0
# 1       10.34  1.66    Male     No  Sun  Dinner   3.0     1.0
# 2       21.01  3.50    Male     No  Sun  Dinner   3.0     1.0
# 3       23.68  3.31    Male     No  Sun  Dinner   2.0     1.0
# 4         NaN  3.61  Female     No  Sun  Dinner   4.0     0.0

# Define recode_gender()
def recode_gender(gender):

    # Return 0 if gender is 'Female'
    if gender == 'Female':
        return 0
    
    # Return 1 if gender is 'Male'    
    elif gender == 'Male':
        return 1
    
    # Return np.nan    
    else:
        return np.nan

# Apply the function to the sex column
tips['recode'] = tips.sex.apply(recode_gender)

# Print the first five rows of tips
print(tips.head(5))

       # total_bill   tip     sex smoker  day    time  size  recode
    # 0       16.99  1.01  Female     No  Sun     NaN   2.0     0.0
    # 1       10.34  1.66    Male     No  Sun  Dinner   3.0     1.0
    # 2       21.01  3.50    Male     No  Sun  Dinner   3.0     1.0
    # 3       23.68  3.31    Male     No  Sun  Dinner   2.0     1.0
    # 4         NaN  3.61  Female     No  Sun  Dinner   4.0     0.0
	
############################################

# Two possibillities to remove something from a column
# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])

# Print the head of tips
print(tips.head())

############################################

# Drop the duplicates: tracks_no_duplicates
tracks_no_duplicates = tracks.drop_duplicates()

# Print info of tracks
print(tracks_no_duplicates.info())

############################################

# Fill missing values with mean
# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality.Ozone.mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality.Ozone.fillna(oz_mean)

############################################

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 122 entries, 0 to 121
# Data columns (total 18 columns):
# Date                   122 non-null object
# Day                    122 non-null int64
# Cases_Guinea           122 non-null float64
# Cases_Liberia          122 non-null float64
# Cases_SierraLeone      122 non-null float64

# Assert that there are no missing values
assert ebola.notnull().all().all()

# Assert that all values are >= 0
assert (ebola >= 0).all().all()

############################################


def check_null_or_valid(row_data):
    """Function that takes a row of data,
    drops all missing values,
    and checks if all remaining values are greater than or equal to 0
    """
    no_na = row_data.dropna()
    numeric = pd.to_numeric(no_na)
    ge0 = numeric >= 0
    return ge0

# Check whether the first column is 'Life expectancy'
assert g1800s.columns[0] == 'Life expectancy'

# Check whether the values in the row are valid
assert g1800s.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()

# Check that there is only one instance of each country
assert g1800s['Life expectancy'].value_counts()[0] == 1

############################################

# Concatenate the DataFrames column-wise
gapminder = pd.concat([g1800s, g1900s, g2000s],axis=1)

# Print the shape of gapminder
print(gapminder.shape)

# Print the head of gapminder
print(gapminder.head())

############################################

import pandas as pd

# Melt gapminder: gapminder_melt
gapminder_melt = gapminder.melt(id_vars='Life expectancy')

# Rename the columns
gapminder_melt.columns = ['country', 'year', 'life_expectancy']

# Print the head of gapminder_melt
print(gapminder_melt.head())

                     # country  year  life_expectancy
    # 0               Abkhazia  1800              NaN
    # 1            Afghanistan  1800            28.21
    # 2  Akrotiri and Dhekelia  1800              NaN
    # 3                Albania  1800            35.40
    # 4                Algeria  1800            28.82
	
############################################

# Convert the year column to numeric
gapminder.year = pd.to_numeric(gapminder.year)

# Test if country is of type object
assert gapminder.country.dtypes == np.object

# Test if year is of type int64
assert gapminder.year.dtypes == np.int64

# Test if life_expectancy is of type float64
assert gapminder.life_expectancy.dtype == np.float64

############################################

# Create the series of countries: countries
countries = gapminder['country']

# Drop all the duplicates from countries
countries = countries.drop_duplicates()

# Write the regular expression: pattern
pattern = '^[A-Za-z\.\s]*$'

# Create the Boolean vector: mask
mask = countries.str.contains(pattern)

# Invert the mask: mask_inverse
mask_inverse = ~mask

# Subset countries using mask_inverse: invalid_countries
invalid_countries = countries.loc[mask_inverse]

# Print invalid_countries
print(invalid_countries)

############################################

# Assert that country does not contain any missing values
assert pd.notnull(gapminder.country).all()

# Assert that year does not contain any missing values
assert pd.notnull(gapminder.year).all()

# Drop the missing values
gapminder = gapminder.dropna()

# Print the shape of gapminder
print(gapminder.shape)

############################################

gapminder.head()
# Out[1]: 
               # country  year  life_expectancy
# 1          Afghanistan  1800            28.21
# 3              Albania  1800            35.40
# 4              Algeria  1800            28.82
# 7               Angola  1800            26.98
# 9  Antigua and Barbuda  1800            33.54

# Add first subplot
plt.subplot(2, 1, 1) 

# Create a histogram of life_expectancy
gapminder.life_expectancy.plot(kind='hist')

# Group gapminder: gapminder_agg
gapminder_agg = gapminder.groupby('year')['life_expectancy'].mean()

# Print the head of gapminder_agg
print(gapminder_agg.head())

# Print the tail of gapminder_agg
print(gapminder_agg.tail())

# Add second subplot
plt.subplot(2, 1, 2)

# Create a line plot of life expectancy per year
gapminder_agg.plot()

# Add title and specify axis labels
plt.title('Life expectancy over the years')
plt.ylabel('Life expectancy')
plt.xlabel('Year')

# Display the plots
plt.tight_layout()
plt.show()

# Save both DataFrames to csv files
gapminder.to_csv('gapminder.csv')
gapminder_agg.to_csv('gapminder_agg.csv')

gapminder_agg.head()
# Out[3]: 
# year
# 1800    31.486020
# 1801    31.448905
# 1802    31.463483
# 1803    31.377413
# 1804    31.446318

############################################

tb.head()
# Out[1]: 
  # country  year  m014  m1524  m2534 ...  f3544  f4554  f5564   f65  fu
# 0      AD  2000   0.0    0.0    1.0 ...    NaN    NaN    NaN   NaN NaN
# 1      AE  2000   2.0    4.0    4.0 ...    3.0    0.0    0.0   4.0 NaN
# 2      AF  2000  52.0  228.0  183.0 ...  339.0  205.0   99.0  36.0 NaN
# 3      AG  2000   0.0    0.0    0.0 ...    0.0    0.0    0.0   0.0 NaN
# 4      AL  2000   2.0   19.0   21.0 ...    8.0    8.0    5.0  11.0 NaN

# Melt tb: tb_melt
tb_melt = pd.melt(frame=tb, id_vars=['country', 'year'])

tb_melt.head()
# Out[3]: 
  # country  year variable  value
# 0      AD  2000     m014    0.0
# 1      AE  2000     m014    2.0
# 2      AF  2000     m014   52.0
# 3      AG  2000     m014    0.0
# 4      AL  2000     m014    2.0

# Create the 'gender' column
tb_melt['gender'] = tb_melt.variable.str[0]

# Create the 'age_group' column
tb_melt['age_group'] = tb_melt.variable.str[1:]

# Print the head of tb_melt
print(tb_melt.head())

      # country  year variable  value gender age_group
    # 0      AD  2000     m014    0.0      m       014
    # 1      AE  2000     m014    2.0      m       014
    # 2      AF  2000     m014   52.0      m       014
    # 3      AG  2000     m014    0.0      m       014
    # 4      AL  2000     m014    2.0      m       014
	
############################################

# Easy list comprehension
# Print original and new data containers
[print(x, 'has type', type(eval(x))) for x in ['np_vals', 'np_vals_log10', 'df', 'df_log10']]

############################################

# Build a dataframe out of lists by zip
# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys,list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)

############################################

# Build a DataFrame with broadcasting
# Make a string with the value 'PA': state
state = 'PA'

# Construct a dictionary: data
data = {'state':state, 'city':cities}

# Construct a DataFrame from dictionary data: df
df = pd.DataFrame(data)

############################################

# Ignore commented lines when reading in
# Read the raw file as-is: df1
df1 = pd.read_csv(file_messy)

# Print the output of df1.head()
print(df1.head())

# Read in the file with the correct parameters: df2
df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')

############################################

# Basic plotting in DataFrame, Pandas
# Create a plot with color='red'
df.plot(color='red')

# Add a title
plt.title('Temperature in Austin')

# Specify the x-axis label
plt.xlabel('Hours since midnight August 1, 2010')

# Specify the y-axis label
plt.ylabel('Temperature (degrees F)')

# Display the plot
plt.show()

############################################

# Setting Subplot is so convenient
# Plot all columns as subplots
df.plot(subplots=True)
plt.show()

# Plot just the Dew Point data
column_list1 = ['Dew Point (deg F)']
df[column_list1].plot()
plt.show()

############################################

# Multiple axis
# Create a list of y-axis column names: y_columns
y_columns = ['AAPL','IBM']

# Generate a line plot
df.plot(x='Month', y=y_columns)

# Add the title
plt.title('Monthly stock prices')

# Add the y-axis label
plt.ylabel('Price ($US)')

# Display the plot
plt.show()

############################################

# Generate a parametric scatter plot
df.plot(kind='scatter', x='hp', y='mpg', s=sizes)

############################################

# Another nice way of doing subplot in Pandas
# Make a list of the column names to be plotted: cols
cols = ['weight' ,'mpg']

# Generate the box plots
df[cols].plot(kind='box', subplots=True)

# Display the plot
plt.show()

############################################

# Doing Subplot and manually arranging
# Plot formatting with fig, axes
# Remember, when plotting the PDF, you need to specify normed=True in your call to .hist(), and when plotting the CDF, you need to specify cumulative=True in addition to normed=True.
# This formats the plots such that they appear on separate rows
fig, axes = plt.subplots(nrows=2, ncols=1)

# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(0,.3))
plt.show()

# Plot the CDF
df.fraction.plot(ax=axes[1], kind='hist', normed=True, cumulative=True, bins=30, range=(0,.3))
plt.show()

############################################

# Print the minimum value of the Engineering column
print(df.Engineering.min())

# Print the maximum value of the Engineering column
print(df.Engineering.max())

# Construct the mean percentage per year: mean
mean = df.mean(axis='columns')

# Plot the average percentage per year
mean.plot()

############################################

# Quantile and Boxplot over timeseries
# Print the number of countries reported in 2015
print(df['2015'].count() )

# Print the 5th and 95th percentiles
print(df.quantile([0.05, 0.95]) )

# Generate a box plot
years = ['1800','1850','1900','1950','2000']
df[years].plot(kind='box')
plt.show()

############################################

# Extracting groups and compare the difference to global set
# Compute the global mean and global standard deviation: global_mean, global_std
global_mean = df.mean()
global_std = df.std()

# Filter the US population from the origin column: us
us = df[df.origin == 'US' ]

# Compute the US mean and US standard deviation: us_mean, us_std
us_mean = us.mean()
us_std = us.std()

# Print the differences
print(us_mean - global_mean)
print(us_std - global_std)

############################################

# Using boolean operation to conduct multiple boxplots
titanic.iloc[:4,:4]
# Out[1]: 
   # pclass  survived                                  name     sex
# 0       1         1         Allen, Miss. Elisabeth Walton  female
# 1       1         1        Allison, Master. Hudson Trevor    male
# 2       1         0          Allison, Miss. Helen Loraine  female
# 3       1         0  Allison, Mr. Hudson Joshua Creighton    male

# Display the box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(nrows=3, ncols=1)

# Generate a box plot of the fare prices for the First passenger class
titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')

# Generate a box plot of the fare prices for the Second passenger class
titanic.loc[titanic['pclass'] == 2].plot(ax=axes[1], y='fare', kind='box')

# Generate a box plot of the fare prices for the Third passenger class
titanic.loc[titanic['pclass'] == 3].plot(ax=axes[2], y='fare', kind='box')

# Display the plot
plt.show()

############################################

# Correct importing and parsing of Date/Time info with autoindexing
df3 = pd.read_csv(filename, index_col='Date', parse_dates=True)

############################################

# Convert the output time format
# Prepare a format string: time_format
time_format = '%Y-%m-%d %H:%M'

# Convert date_list into a datetime object: my_datetimes
my_datetimes = pd.to_datetime(date_list, format=time_format)  

# Construct a pandas Series using temperature_list and my_datetimes: time_series
time_series = pd.Series(temperature_list, index=my_datetimes)

############################################

# Extract the hour from 9pm to 10pm on '2010-10-11': ts1
ts1 = ts0.loc['2010-10-11 21:00:00':'2010-10-11 22:00:00']

# Extract '2010-07-04' from ts0: ts2
ts2 = ts0.loc['2010-07-04']

# Extract data from '2010-12-15' to '2010-12-31': ts3
ts3 = ts0.loc['2010-12-15':'2010-12-31']

############################################

# dir() will give you the list of in scope variables:
# globals() will give you a dictionary of global variables
# locals() will give you a dictionary of local variables
# %who will give you the iPython local variables

############################################

# Downsampling
# Downsample to 6 hour data and aggregate by mean: df1
df1 = df.Temperature.resample('6h').mean()

# Downsample to daily data and count the number of data points: df2
df2 = df.Temperature.resample('D').count()

############################################

# Resampling Method in Downsampling
# Extract temperature data for August: august
august = df.loc['2010-August','Temperature']

# Downsample to obtain only the daily highest temperatures in August: august_highs
august_highs = august.resample('1D').max()

# Extract temperature data for February: february
february = df.loc['2010-02','Temperature']

# Downsample to obtain the daily lowest temperatures in February: february_lows
february_lows = february.resample('1D').min()

############################################

# Resampling, Window-Function, mean, moving average
# Extract data from 2010-Aug-01 to 2010-Aug-15: unsmoothed
unsmoothed = df['Temperature']['2010-08-01':'2010-08-15']  # Data indexing, slecting like this is convenient

# Apply a rolling mean with a 24 hour window: smoothed
smoothed = unsmoothed.rolling(24).mean()

# Create a new DataFrame with columns smoothed and unsmoothed: august
august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})

# Plot both smoothed and unsmoothed data using august.plot().
august.plot()
plt.show()

############################################

# Moving average on resampled data
# Extract the August 2010 data: august
august = df['Temperature']['2010-08']

# Resample to daily data, aggregating by max: daily_highs
daily_highs = august.resample('1D').max()

# Use a rolling 7-day window with method chaining to smooth the daily high temperatures in August
daily_highs_smoothed = daily_highs.rolling(7).mean()
print(daily_highs_smoothed)

############################################

# Strip extra whitespace from the column names: df.columns
df.columns = df.columns.str.strip()

# Extract data for which the destination airport is Dallas: dallas
dallas = df['Destination Airport'].str.contains('DAL')

# Compute the total number of Dallas departures each day: daily_departures
daily_departures = dallas.resample('1D').sum()

# Generate the summary statistics for daily Dallas departures: stats
stats = daily_departures.describe()

############################################

