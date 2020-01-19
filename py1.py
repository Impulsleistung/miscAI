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

# Alter the timebase, not the data. Interpolation of data (ts2) to the timebase of ts1 . 
# Data of ts1 remains untouched. ts2 gets only interpolated to the new timebase
# Reset the index of ts2 to ts1, and then use linear interpolation to fill in the NaNs: ts2_interp
ts2_interp = ts2.reindex(index=ts1.index).interpolate(how='linear')

# Compute the absolute difference of ts1 and ts2_interp: differences 
differences = np.abs(ts2_interp-ts1)

# Generate and print summary statistics of the differences
print(differences.describe())

############################################

# Setting and changing timezones
# Build a Boolean mask to filter for the 'LAX' departure flights: mask
mask = df['Destination Airport'] == 'LAX'

# Use the mask to subset the data: la
la = df[mask]

# Combine two columns of data to create a datetime series: times_tz_none 
times_tz_none = pd.to_datetime(la['Date (MM/DD/YYYY)'] + ' ' + la['Wheels-off Time'] )

# Localize the time to US/Central: times_tz_central
times_tz_central = times_tz_none.dt.tz_localize('US/Central')

# Convert the datetimes from US/Central to US/Pacific
times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')

############################################

# Set index and plotting
# Convert the 'Date' column into a collection of datetime objects: df.Date
df.Date = pd.to_datetime(df.Date)

# Set the index to be the converted 'Date' column
df.set_index('Date', inplace=True)

# Re-plot the DataFrame to see that the axis is now datetime aware!
df.plot()
plt.show()

############################################

# Split on the comma to create a list: column_labels_list
column_labels_list = column_labels.split(',')

# Assign the new column labels to the DataFrame: df.columns
df.columns = column_labels_list

# Remove the appropriate columns: df_dropped
df_dropped = df.drop(list_to_drop, axis='columns')

# Print the output of df_dropped.head()
print(df_dropped.head())

############################################

# Convert the date column to string: df_dropped['date']
df_dropped['date'] = df_dropped['date'].astype(str)

# Padding (adding zeros) by regular expession and format-function
# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))

# Concatenate the new date and Time columns: date_string
date_string = df_dropped.date+df_dropped.Time

# Convert the date_string Series to datetime: date_times
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')

# Set the index to be the new date_times container: df_clean
df_clean = df_dropped.set_index(date_times)

# Print the output of df_clean.head()
print(df_clean.head())

############################################

# interpret missing data as NaN. The pandas function pd.to_numeric() is ideal for this purpose: It converts a Series of values to floating-point values. Furthermore, by specifying the keyword argument errors='coerce', you can force strings like 'M' to be interpreted as NaN.

# Print the dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-06-20 08:00:00':'2011-06-20 09:00:00', 'dry_bulb_faren'])

# Convert the dry_bulb_faren column to numeric values: df_clean['dry_bulb_faren']
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')

# Print the transformed dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-06-20 08:00:00':'2011-06-20 09:00:00', 'dry_bulb_faren'])

# Convert the wind_speed and dew_point_faren columns to numeric values
df_clean['wind_speed'] = pd.to_numeric(df_clean['wind_speed'], errors='coerce')
df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'], errors='coerce')

############################################

# Print the median of the dry_bulb_faren column
print(df_clean['dry_bulb_faren'].median())

# Print the median of the dry_bulb_faren column for the time range '2011-Apr':'2011-Jun'
print(df_clean.loc['2011-Apr':'2011-Jun', 'dry_bulb_faren'].median())

############################################

# Downsampling and comparing two time series measurements with different timebase
# Downsample df_clean by day and aggregate by mean: daily_mean_2011
daily_mean_2011 = df_clean.resample('1D').mean()

# Extract the dry_bulb_faren column from daily_mean_2011 using .values: daily_temp_2011
daily_temp_2011 = daily_mean_2011['dry_bulb_faren'].values

# Downsample df_climate by day and aggregate by mean: daily_climate
daily_climate = df_climate.resample('1D').mean()

# Extract the Temperature column from daily_climate using .reset_index(): daily_temp_climate
daily_temp_climate = daily_climate.reset_index()['Temperature']

# Compute the difference between the two arrays and print the mean difference
difference = daily_temp_2011 - daily_temp_climate
print(difference.mean())

############################################

# Mapping Pandas and resample
# Using df_clean, when does sky_condition contain 'OVC'?
is_sky_overcast = df_clean['sky_condition'].str.contains('OVC')

# Filter df_clean using is_sky_overcast
overcast = df_clean.loc[is_sky_overcast]

# Resample overcast by day then calculate the max
overcast_daily_max = overcast.resample('1D').max()

# See the result
overcast_daily_max.head()

############################################

# Resampling several columns at once and automatic subplot them
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Select the visibility and dry_bulb_faren columns and resample them: weekly_mean
weekly_mean = df_clean[['visibility' ,'dry_bulb_faren']].resample('1W').mean()

# Print the output of weekly_mean.corr()
print(weekly_mean.corr())

# Plot weekly_mean with subplots=True
weekly_mean.plot(subplots=True)
plt.show()

############################################

# Using masking and the resample generator for generating daily statistics
is_sky_clear = df_clean['sky_condition'] == 'CLR'
resampled = is_sky_clear.resample('D')
sunny_hours = resampled.sum()
total_hours = resampled.count()
sunny_fraction = sunny_hours / total_hours

# Make a box plot of sunny_fraction
sunny_fraction.plot(kind='box')
plt.show()

############################################

# CDF plot to outline the overshot in temperature compared to previous year
# Using .loc to filter for overshot temperature
# Extract the maximum temperature in August 2010 from df_climate: august_max
august_max = df_climate.loc['2010-08','Temperature'].max()
print(august_max)

# Resample August 2011 temps in df_clean by day & aggregate the max value: august_2011
august_2011 = df_clean.loc['2011-08','dry_bulb_faren'].resample('D').max()

# Filter for days in august_2011 where the value exceeds august_max: august_2011_high
august_2011_high = august_2011.loc[august_2011> august_max]

# Construct a CDF of august_2011_high
august_2011_high.plot(kind='hist', normed=True, cumulative=True, bins=25)

# Display the plot
plt.show()

############################################

df[['colname']] # Return Dataframe
df['colname']   # Return Series

############################################

# Forward and backward extraction of rows in dataframe
# Slice the row labels 'Perry' to 'Potter': p_counties
p_counties = election.loc['Perry':'Potter']

# Print the p_counties DataFrame
print(p_counties)

# Slice the row labels 'Potter' to 'Perry' in reverse order: p_counties_rev
p_counties_rev = election.loc['Potter':'Perry':-1]

# Print the p_counties_rev DataFrame
print(p_counties_rev)

############################################

#################################

# Dropping rows with ANY and ALL
# 
# Select the 'age' and 'cabin' columns: df
df = titanic[['age', 'cabin']]

# Print the shape of df
print(df.shape)

# Drop rows in df with how='any' and print the shape
print(df.dropna(how='any').shape)

# Drop rows in df with how='all' and print the shape
print(df.dropna(how='all').shape)

# Drop columns in titanic with less than 1000 non-missing values
print(titanic.dropna(thresh=1000, axis='columns').info())

#################################

# Using apply on several columns and rename those columns
# Write a function to convert degrees Fahrenheit to degrees Celsius: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)

# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = weather[['Mean TemperatureF','Mean Dew PointF']].apply(to_celsius)

# Reassign the column labels of df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']

# Print the output of df_celsius.head()
print(df_celsius.head())

#################################

# Mapping of values to create a new column
# Create the dictionary: red_vs_blue
red_vs_blue = {'Obama':'blue' , 'Romney':'red'}

# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election.winner.map(red_vs_blue)

# Print the output of election.head()
print(election.head())


# <script.py> output:
              # state   total      Obama     Romney  winner  voters color
    # county                                                             
    # Adams        PA   41973  35.482334  63.112001  Romney   61156   red
    # Allegheny    PA  614671  56.640219  42.185820   Obama  924351  blue
    # Armstrong    PA   28322  30.696985  67.901278  Romney   42147   red
    # Beaver       PA   80015  46.032619  52.637630  Romney  115157   red
    # Bedford      PA   21444  22.057452  76.986570  Romney   32189   red

#################################

# Calculation of the Z-Score and adding a new column
# Import zscore from scipy.stats
from scipy.stats import zscore

# Call zscore with election['turnout'] as input: turnout_zscore
turnout_zscore = zscore(election['turnout'])

# Print the type of turnout_zscore
print(type(turnout_zscore))

# Assign turnout_zscore to a new column: election['turnout_zscore']
election['turnout_zscore'] = turnout_zscore

# Print the output of election.head()
print(election.head())

#################################

# Alter the index by using of list comprehension
# Create the list of new indexes: new_idx
new_idx = [str.upper(i) for i in sales.index]

# Assign new_idx to sales.index
sales.index = new_idx

# Print the sales DataFrame
print(sales)

#################################

# Assign the string 'MONTHS' to sales.index.name
sales.index.name = 'MONTHS'
# Assign the string 'PRODUCTS' to sales.columns.name 
sales.columns.name = 'PRODUCS'

# Print the sales dataframe again
print(sales)

#################################

# MultiIndex define based on columns
# Set the index to be the columns ['state', 'month']: sales
sales = sales.set_index(['state', 'month'])

# Sort the MultiIndex: sales
sales = sales.sort_index()

#################################

# Operating the MultiIndex in three different ways

             # eggs  salt  spam
# state month                  
# CA    1        47  12.0    17
      # 2       110  50.0    31
# NY    1       221  89.0    72
      # 2        77  87.0    20
# TX    1       132   NaN    52

# Look up data for NY in month 1 in sales: NY_month1
NY_month1 = sales.loc[('NY',1),:]

# Look up data for CA and TX in month 2: CA_TX_month2
CA_TX_month2 = sales.loc[(['CA','TX'],2),:]

# Access the inner month index and look up data for all states in month 2: all_month2
all_month2 = sales.loc[(slice(None),2),:]

#################################

# Data pivoting. Columns which are not pivoted will be dropped

  # weekday    city  visitors  signups
# 0     Sun  Austin       139        7
# 1     Sun  Dallas       237       12
# 2     Mon  Austin       326        3
# 3     Mon  Dallas       456        5

# Pivot the users DataFrame: visitors_pivot
visitors_pivot = users.pivot(index='weekday',columns='city',values='visitors')

    # city     Austin  Dallas
    # weekday                
    # Mon         326     456
    # Sun         139     237
	
#################################

# Hirarchical pivoting

  # weekday    city  visitors  signups
# 0     Sun  Austin       139        7
# 1     Sun  Dallas       237       12
# 2     Mon  Austin       326        3
# 3     Mon  Dallas       456        5

# Pivot users with signups indexed by weekday and city: signups_pivot
signups_pivot = users.pivot(index='weekday',columns='city', values='signups')

    # city     Austin  Dallas
    # weekday                
    # Mon           3       5
    # Sun           7      12

# Pivot users pivoted by both signups and visitors: pivot
pivot = users.pivot(index='weekday', columns='city')

            # visitors        signups       
    # city      Austin Dallas  Austin Dallas
    # weekday                               
    # Mon          326    456       3      5
    # Sun          139    237       7     12

#################################

# The following dataframe has two levels pivoted
                visitors  signups
city   weekday                   
Austin Mon           326        3
       Sun           139        7
Dallas Mon           456        5
       Sun           237       12
	   
# The city shall not be a sub-index but a sub-column, so unstack	   
bycity = users.unstack('city')
            visitors        signups       
    city      Austin Dallas  Austin Dallas
    weekday                               
    Mon          326    456       3      5
    Sun          139    237       7     12
	
# And reverse: The city shall become a sub-index
# Stack bycity by 'city' and print it
print(bycity.stack('city'))

                    visitors  signups
    weekday city                     
    Mon     Austin       326        3
            Dallas       456        5
    Sun     Austin       139        7
            Dallas       237       12
			
#################################

# Continuing from the previous exercise, you will now use .swaplevel(0, 1) to flip the index levels. Note they won't be sorted. To sort them, you will have to follow up with a .sort_index().
# Stack 'city' back into the index of bycity: newusers
newusers = bycity.stack('city')

# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0,1)

# Print newusers and verify that the index is not sorted
print(newusers)

# Sort the index of newusers: newusers
newusers = newusers.sort_index()

# Print newusers and verify that the index is now sorted
print(newusers)

# Verify that the new DataFrame is equal to the original
print(newusers.equals(users))

#################################

# Melting in Python
  weekday    city  visitors  signups
0     Sun  Austin       139        7
1     Sun  Dallas       237       12
2     Mon  Austin       326        3
3     Mon  Dallas       456        5

# The 
# id_vars		= column you want to keep as index
# value_vars	= The column(s) you want to reference to the index
pd.melt(df, id_vars =['Name'], value_vars =['Course']) 

#################################

# Pivot dataframe as a table with one index column and all others are derived from a second column

  weekday    city  visitors  signups
0     Sun  Austin       139        7
1     Sun  Dallas       237       12
2     Mon  Austin       326        3
3     Mon  Dallas       456        5

by_city_day = users.pivot_table(index='weekday',columns='city' )

            signups        visitors       
    city     Austin Dallas   Austin Dallas
    weekday                               
    Mon           3      5      326    456
    Sun           7     12      139    237
	
#################################

# Count entries by specified index by using pivot_table
  weekday    city  visitors  signups
0     Sun  Austin       139        7
1     Sun  Dallas       237       12
2     Mon  Austin       326        3
3     Mon  Dallas       456        5


count_by_weekday1 = users.pivot_table(index='weekday', aggfunc='count')


             city  signups  visitors
    weekday                         
    Mon         2        2         2
    Sun         2        2         2

#################################

# Add summary to pivot table

users.pivot_table(index='weekday',aggfunc=sum, margins=True)

             signups  visitors
    weekday                   
    Mon            8       782
    Sun           19       376
    All           27      1158
	
#################################

# Read life_fname into a DataFrame: life
life = pd.read_csv(life_fname, index_col='Country')
# Read regions_fname into a DataFrame: regions
regions = pd.read_csv(regions_fname, index_col='Country')

# The life - expectancy
               1964    1965    1966    1967
Country                                    
Afghanistan  33.639  34.152  34.662  35.170
Albania      65.475  65.863  66.122  66.316
Algeria      47.953  48.389  48.806  49.205
Angola       34.604  35.007  35.410  35.816

# The region
Country                                        
Afghanistan                          South Asia
Albania                   Europe & Central Asia
Algeria              Middle East & North Africa
Angola                       Sub-Saharan Africa
Antigua and Barbuda                     America
			  

# Group life by regions['region']: life_by_region
life_by_region = life.groupby(regions['region'])
print(life_by_region['2010'].mean())

    America                       74.037350
    East Asia & Pacific           73.405750
    Europe & Central Asia         75.656387
    Middle East & North Africa    72.805333
    South Asia                    68.189750
    Sub-Saharan Africa            57.575080
			  
#################################

# Use of aggregate in combination with grouping
In [4]: sales.groupby('city')[['bread','butter']].agg(['max','sum'])
Out[4]:
		bread 	butter
		max sum max sum
city
Austin	326	465	70	90
Dallas	456	693	98	143

#################################

# Groupby and aggregate on the titanic data with multiple aggregates
In [6]: titanic[['age','fare','pclass']].head()
Out[6]: 
     age      fare  pclass
0  29.00  211.3375       1
1   0.92  151.5500       1
2   2.00  151.5500       1
3  30.00  151.5500       1
4  25.00  151.5500       1


# Group titanic by 'pclass': by_class
by_class = titanic.groupby('pclass')

# Select 'age' and 'fare'
by_class_sub = by_class[['age','fare']]

# Aggregate by_class_sub by 'max' and 'median': aggregated
aggregated = by_class_sub.agg(['max','median'])


In [8]: aggregated
Out[8]: 
         age             fare         
         max median       max   median
pclass                                
1       80.0   39.0  512.3292  60.0000
2       70.0   29.0   73.5000  15.0458
3       74.0   24.0   69.5500   8.0500

# Important notice: When to print the max in the age column a tuple has to be adressed
# Print the maximum age in each class
print(aggregated.loc[:, ('age','max')])

# Print the median fare in each class
print(aggregated.loc[:, ('fare','median')] )

#################################

# Unsorted data should be sorted upfront when imported
gapminder = pd.read_csv('gapminder.csv', index_col=['Year','region', 'Country']).sort_index()

In [3]: gapminder.head()
Out[3]: 
                             fertility    life  population  child_mortality     gdp
Year region     Country                                                            
1964 South Asia Afghanistan      7.671  33.639  10474903.0            339.7  1182.0
1965 South Asia Afghanistan      7.671  34.152  10697983.0            334.1  1182.0
1966 South Asia Afghanistan      7.671  34.662  10927724.0            328.7  1168.0
1967 South Asia Afghanistan      7.671  35.170  11163656.0            323.3  1173.0

# Group gapminder by 'Year' and 'region': by_year_region
by_year_region = gapminder.groupby(level=['Year','region'])

# Define the function to compute spread: spread
def spread(series):
    return series.max() - series.min()

# Create the dictionary: aggregator
aggregator = {'population':'sum', 'child_mortality':'mean', 'gdp':spread}

# Aggregate by_year_region using the dictionary: aggregated
aggregated = by_year_region.agg(aggregator)

# Print the last 6 entries of aggregated 
print(aggregated.tail(6))

                                       population  child_mortality       gdp
    Year region                                                             
    2013 America                     9.629087e+08        17.745833   49634.0
         East Asia & Pacific         2.244209e+09        22.285714  134744.0
         Europe & Central Asia       8.968788e+08         9.831875   86418.0
         Middle East & North Africa  4.030504e+08        20.221500  128676.0
         South Asia                  1.701241e+09        46.287500   11469.0
         Sub-Saharan Africa          9.205996e+08        76.944490   32035.0
		 
#################################

# How many Units are sold by day?
# The index is the timebase and it has to be grouped
# Read file: sales
sales = pd.read_csv('sales.csv',index_col='Date', parse_dates=True)

                             Company   Product  Units
Date                                                 
2015-02-02 08:30:00            Hooli  Software      3
2015-02-02 21:00:00        Mediacore  Hardware      9
2015-02-03 14:00:00          Initech  Software     13
2015-02-04 15:30:00        Streeplex  Software     13
2015-02-04 22:00:00  Acme Coporation  Hardware     14

# Create a groupby object: by_day
# %a - abbreviated weekday name
by_day = sales.groupby(sales.index.strftime('%a'))

# Create sum: units_sum
units_sum = by_day['Units'].sum()

# Print units_sum
print(units_sum)

<script.py> output:
    Mon    48
    Sat     7
    Thu    59
    Tue    13
    Wed    48
	
#################################

# There are two major differences between the transform and apply groupby methods.

# Input:
	# apply implicitly passes all the columns for each group as a DataFrame to the custom function.
	# while transform passes each column for each group individually as a Series to the custom function.
# Output:
	# The custom function passed to apply can return a scalar, or a Series or DataFrame (or numpy array or even list).
	# The custom function passed to transform must return a sequence (a one dimensional Series, array or list) the same length as the group.

# Import zscore from scipy.stats.
# Group gapminder_2010 by 'region' and transform the ['life','fertility'] columns by zscore.
# Construct a boolean Series of the bitwise or between standardized['life'] < -3 and standardized['fertility'] > 3.
# Filter gapminder_2010 using .loc[] and the outliers Boolean Series. Save the result as gm_outliers.

# Import zscore
from scipy.stats import zscore

# Group gapminder_2010: standardized
standardized = gapminder_2010.groupby('region')['life','fertility'].transform(zscore)

# Construct a Boolean Series to identify outliers: outliers
outliers = (standardized['life'] < -3) | (standardized['fertility'] > 3)

# Filter gapminder_2010 by the outliers: gm_outliers
gm_outliers = gapminder_2010.loc[outliers]

# Print gm_outliers
print(gm_outliers)

#################################

# Fill missing values by assumption/impute, using the transform function
# Create a groupby object: by_sex_class
by_sex_class = titanic.groupby(['sex','pclass'])

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

# Impute age and assign to titanic['age']
titanic.age = by_sex_class['age'].transform(impute_median)

# Print the output of titanic.tail(10)
print(titanic.tail(10))

#################################

# Calculate  the disparity of country to region
def disparity(gr):
    # Compute the spread of gr['gdp']: s
    s = gr['gdp'].max() - gr['gdp'].min()
    # Compute the z-score of gr['gdp'] as (gr['gdp']-gr['gdp'].mean())/gr['gdp'].std(): z
    z = (gr['gdp'] - gr['gdp'].mean())/gr['gdp'].std()
    # Return a DataFrame with the inputs {'z(gdp)':z, 'regional spread(gdp)':s}
    return pd.DataFrame({'z(gdp)':z , 'regional spread(gdp)':s})
	
# Group gapminder_2010 by 'region': regional
regional = gapminder_2010.groupby('region')

# Apply the disparity function on regional: reg_disp
reg_disp = regional.apply(disparity)

# Print the disparity of 'United States', 'United Kingdom', and 'China'
print(reg_disp.loc[['United States','United Kingdom','China']])

#################################

# You can use groupby with the .filter() method to remove whole groups of rows from a DataFrame based on a boolean condition.

                             Company   Product  Units
Date                                                 
2015-02-02 08:30:00            Hooli  Software      3
2015-02-02 21:00:00        Mediacore  Hardware      9
2015-02-03 14:00:00          Initech  Software     13
2015-02-04 15:30:00        Streeplex  Software     13
2015-02-04 22:00:00  Acme Coporation  Hardware     14

# In this exercise, you'll take the February sales data and remove entries from companies that purchased less than or equal to 35 Units in the whole month.

# Read the CSV file into a DataFrame: sales
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

# Group sales by 'Company': by_company
by_company = sales.groupby('Company')

	# Compute the sum of the 'Units' of by_company: by_com_sum
	by_com_sum = by_company['Units'].sum()
	print(by_com_sum)

# We are only interested in those companies who sell more than 35 Units
# Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g:g['Units'].sum()>35)
print(by_com_filt)

#################################

# your job is to investigate survival rates of passengers on the Titanic by 'age' and 'pclass'. In particular, the goal is to find out what fraction of children under 10 survived in each 'pclass'. You'll do this by first creating a boolean array where True is passengers under 10 years old and False is passengers over 10. You'll use .map() to change these values to strings.

# Finally, you'll group by the under 10 series and the 'pclass' column and aggregate the 'survived' column. The 'survived' column has the value 1 if the passenger survived and 0 otherwise. The mean of the 'survived' column is the fraction of passengers who lived.
# Create a Boolean Series of titanic['age'] < 10 and call .map with {True:'under 10', False:'over 10'}.
# Group titanic by the under10 Series and then compute and print the mean of the 'survived' column.
# Group titanic by the under10 Series as well as the 'pclass' column and then compute and print the mean of the 'survived' column.

# Create the Boolean Series: under10
under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})

# Group by under10 and compute the survival rate
survived_mean_1 = titanic.groupby(under10)['survived'].mean()
print(survived_mean_1)

# Group by under10 and pclass and compute the survival rate
survived_mean_2 = titanic.groupby([under10,'pclass'])['survived'].mean()
print(survived_mean_2)

#################################

# Olympic dataset, count medals by country and sort
# Select the 'NOC' column of medals: country_names
country_names = medals['NOC']

# Count the number of medals won by each country: medal_counts
medal_counts = country_names.value_counts()

# Print top 15 countries ranked by medals
print(medal_counts.head(15))

#################################

# Pivot table of Olympic dataset to list the medals by their category and country
In [1]: medals.head()
Out[1]: 
     City  Edition     Sport Discipline             Athlete  NOC Gender                       Event Event_gender   Medal
0  Athens     1896  Aquatics   Swimming       HAJOS, Alfred  HUN    Men              100m freestyle            M    Gold
1  Athens     1896  Aquatics   Swimming    HERSCHMANN, Otto  AUT    Men              100m freestyle            M  Silver
2  Athens     1896  Aquatics   Swimming   DRIVAS, Dimitrios  GRE    Men  100m freestyle for sailors            M  Bronze
3  Athens     1896  Aquatics   Swimming  MALOKINIS, Ioannis  GRE    Men  100m freestyle for sailors            M    Gold
4  Athens     1896  Aquatics   Swimming  CHASAPIS, Spiridon  GRE    Men  100m freestyle for sailors            M  Silver

counted = medals.pivot_table(index='NOC',aggfunc='count',values='Athlete',columns='Medal')

In [2]: counted
Out[2]: 
Medal  Bronze    Gold  Silver
NOC                          
AFG       1.0     NaN     NaN
AHO       NaN     NaN     1.0
ALG       8.0     4.0     2.0
ANZ       5.0    20.0     4.0
ARG      88.0    68.0    83.0
ARM       7.0     1.0     1.0
AUS     413.0   293.0   369.0
AUT      44.0    21.0    81.0

# Sum along the column to give the total by row
counted['totals'] = counted.sum(axis='columns')

In [4]: counted
Out[4]: 
Medal  Bronze    Gold  Silver  totals
NOC                                  
AFG       1.0     NaN     NaN     1.0
AHO       NaN     NaN     1.0     1.0
ALG       8.0     4.0     2.0    14.0
ANZ       5.0    20.0     4.0    29.0
ARG      88.0    68.0    83.0   239.0

counted = counted.sort_values('totals', ascending=False)

<script.py> output:
    Medal  Bronze    Gold  Silver  totals
    NOC                                  
    USA    1052.0  2088.0  1195.0  4335.0
    URS     584.0   838.0   627.0  2049.0
    GBR     505.0   498.0   591.0  1594.0
    FRA     475.0   378.0   461.0  1314.0
    ITA     374.0   460.0   394.0  1228.0

#################################

# Elegant boolean mapping
# Create the Boolean Series: sus
sus = (medals.Event_gender == 'W') & (medals.Gender == 'Men')

# Create a DataFrame with the suspicious row: suspect
suspect = medals[sus]

# Print suspect
print(suspect)

#################################

In [4]: medals.iloc[:5,:8]
Out[4]: 
     City  Edition     Sport Discipline             Athlete  NOC Gender                       Event
0  Athens     1896  Aquatics   Swimming       HAJOS, Alfred  HUN    Men              100m freestyle
1  Athens     1896  Aquatics   Swimming    HERSCHMANN, Otto  AUT    Men              100m freestyle
2  Athens     1896  Aquatics   Swimming   DRIVAS, Dimitrios  GRE    Men  100m freestyle for sailors
3  Athens     1896  Aquatics   Swimming  MALOKINIS, Ioannis  GRE    Men  100m freestyle for sailors
4  Athens     1896  Aquatics   Swimming  CHASAPIS, Spiridon  GRE    Men  100m freestyle for sailors

# Using medals, create a Boolean Series called during_cold_war that is True when 'Edition' is >= 1952 and <= 1988.
# Using medals, create a Boolean Series called is_usa_urs that is True when 'NOC' is either 'USA' or 'URS'.
# Filter the medals DataFrame using during_cold_war and is_usa_urs to create a new DataFrame called cold_war_medals.
# Group cold_war_medals by 'NOC'.
# Create a Series Nsports from country_grouped using indexing & chained methods:
# Extract the column 'Sport'.
# Use .nunique() to get the number of unique elements in each group;
# Apply .sort_values(ascending=False) to rearrange the Series.

# Overlapping boolean maps 
# Create a Boolean Series that is True when 'Edition' is between 1952 and 1988: during_cold_war
during_cold_war = (medals.Edition >= 1952) & (medals.Edition <= 1988)

# Extract rows for which 'NOC' is either 'USA' or 'URS': is_usa_urs
is_usa_urs = medals.NOC.isin(['USA','URS'])

# Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals
cold_war_medals = medals.loc[during_cold_war & is_usa_urs]

# Group cold_war_medals by 'NOC'
country_grouped = cold_war_medals.groupby('NOC')

# Create Nsports
Nsports = country_grouped['Sport'].nunique().sort_values(ascending=False)

# Print Nsports
print(Nsports)

<script.py> output:
    NOC
    URS    21
    USA    20
    Name: Sport, dtype: int64
	
#################################

# Construct medals_won_by_country using medals.pivot_table().
# - The index should be the years ('Edition') & the columns should be country ('NOC')
# - The values should be 'Athlete' (which captures every medal regardless of kind) & the aggregation method should be 'count' (which captures the total number of medals won).
# - Create cold_war_usa_urs_medals by slicing the pivot table medals_won_by_country. Your slice should contain the editions from years 1952:1988 and only the columns 'USA' & 'URS' from the pivot table.
# - Create the Series most_medals by applying the .idxmax() method to cold_war_usa_urs_medals. Be sure to use axis='columns'.
# - Print the result of applying .value_counts() to most_medals. The result reported gives the number of times each of the USA or the USSR won more Olympic medals in total than the other between 1952 and 1988.

# Create the pivot table: medals_won_by_country
medals_won_by_country = medals.pivot_table(index='Edition',columns='NOC',values='Athlete',aggfunc='count')

# Slice medals_won_by_country: cold_war_usa_urs_medals
cold_war_usa_urs_medals = medals_won_by_country.loc[1952:1988, ['USA','URS']]

# Create most_medals 
most_medals = cold_war_usa_urs_medals.idxmax(axis='columns')

# Print most_medals.value_counts()
print(most_medals.value_counts())

#################################

# Create the DataFrame: usa
usa = medals[medals.NOC=='USA']

# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = usa.groupby(['Edition','Medal']).aggregate('Athlete').count()

In [4]: usa_medals_by_year.head()
Out[4]: 
Edition  Medal 
1896     Bronze     2
         Gold      11
         Silver     7
1900     Bronze    14
         Gold      27
Name: Athlete, dtype: int64

usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

usa_medals_by_year.head()

Medal    Bronze  Gold  Silver
Edition                      
1896          2    11       7
1900         14    27      14
1904        111   146     137
1908         15    34      14
1912         31    45      25

# Nice plotting goes like
# Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot()
plt.show()

# Area is better
# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()

#################################
# Re-Ordering the categories in the right order
# Redefine the 'Medal' column of the DataFrame medals as an ordered categorical. To do this, use pd.Categorical() with three keyword arguments:
# values = medals.Medal.
# categories=['Bronze', 'Silver', 'Gold'].
# ordered=True.

