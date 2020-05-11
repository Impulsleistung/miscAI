# Reading filenames as list in a while loop
## Create the list of file names: filenames
filenames = ['Gold.csv', 'Silver.csv', 'Bronze.csv']

# Create the list of three DataFrames: dataframes
dataframes = []
for filename in filenames:
    dataframes.append(pd.read_csv(filename))

#######################################

# Merge two dataframes with different column names
          city  branch_id state  revenue
0       Austin         10    TX      100
1       Denver         20    CO       83
2  Springfield         30    IL        4
3    Mendocino         47    CA      200

        branch  branch_id state   manager
0       Austin         10    TX  Charlers
1       Denver         20    CO      Joel
2    Mendocino         47    CA     Brett
3  Springfield         31    MO     Sally

combined = pd.merge(revenue, managers, left_on='city', right_on='branch')

<script.py> output:
              city  branch_id_x state_x  revenue       branch  branch_id_y state_y   manager
    0       Austin           10      TX      100       Austin           10      TX  Charlers
    1       Denver           20      CO       83       Denver           20      CO      Joel
    2  Springfield           30      IL        4  Springfield           31      MO     Sally
    3    Mendocino           47      CA      200    Mendocino           47      CA     Brett

#######################################

# Generate extra column with information for more precise merge
# The parameter on= means always inner join
# Add 'state' column to revenue: revenue['state']
revenue['state'] = ['TX','CO','IL','CA']

# Add 'state' column to managers: managers['state']
managers['state'] = ['TX','CO','CA','MO']

# Merge revenue & managers on 'branch_id', 'city', & 'state': combined
combined = pd.merge(revenue, managers, on=['branch_id', 'city', 'state'])

<script.py> output:
            city  branch_id  revenue state   manager
    0     Austin         10      100    TX  Charlers
    1     Denver         20       83    CO      Joel
    2  Mendocino         47      200    CA     Brett
	
#######################################

df.join(df2) # is a left join by default

df.append(df2) # Stacking vert.
pd.concat(df1,df2) # Stacking horizonatal or vertical, 'inner' or 'outer' join, index-based
df.join(df2) # inner, outer, left, right on index
pd.merge([df1,df2]) # all thinkable joins on multiple columns

# Left join explanation
# - Keep all rows of the left df
# - non joining cols of the right get append
# - non matching columns of the left df get filled with nulls

#######################################

revenue
          city  branch_id state  revenue
0       Austin         10    TX      100
1       Denver         20    CO       83
2  Springfield         30    IL        4
3    Mendocino         47    CA      200

managers
        branch  branch_id state   manager
0       Austin         10    TX  Charlers
1       Denver         20    CO      Joel
2    Mendocino         47    CA     Brett
3  Springfield         31    MO     Sally

sales
          city state  units
0    Mendocino    CA      1
1       Denver    CO      4
2       Austin    TX      2
3  Springfield    MO      5
4  Springfield    IL      1

# Missing entries on the left get append when using right join
# on= is the AND criteria for the columns
# Merge revenue and sales: revenue_and_sales
revenue_and_sales = pd.merge(revenue, sales, how='right', on=['city', 'state'])

              city  branch_id state  revenue  units
    0       Austin       10.0    TX    100.0      2
    1       Denver       20.0    CO     83.0      4
    2  Springfield       30.0    IL      4.0      1
    3    Mendocino       47.0    CA    200.0      1
    4  Springfield        NaN    MO      NaN      5

# The columns AND criteria when 2 AND 2 must be matched but have different column names is left_on, right_on
# ALL left entries have to be included -> adds the missing
# Merge sales and managers: sales_and_managers
sales_and_managers = pd.merge(sales, managers, how='left', left_on=['city','state'], right_on=['branch', 'state'])

              city state  units       branch  branch_id   manager
    0    Mendocino    CA      1    Mendocino       47.0     Brett
    1       Denver    CO      4       Denver       20.0      Joel
    2       Austin    TX      2       Austin       10.0  Charlers
    3  Springfield    MO      5  Springfield       31.0     Sally
    4  Springfield    IL      1          NaN        NaN       NaN
	
# Crucial difference:
# ALL right have to be included -> does NOT add the left without a matching column criteria on the right
In [2]: pd.merge(sales, managers, how='right', left_on=['city','state'], right_on=['branch', 'state'])
          city state  units       branch  branch_id   manager
0    Mendocino    CA      1    Mendocino         47     Brett
1       Denver    CO      4       Denver         20      Joel
2       Austin    TX      2       Austin         10  Charlers
3  Springfield    MO      5  Springfield         31     Sally

#######################################

# merge_ordered is equal to join='outer' and sort_by='date'
# The suffixes will expand the non-index column names for unique classification
austin
        date ratings
0 2016-01-01  Cloudy
1 2016-02-08  Cloudy
2 2016-01-17   Sunny

houston
        date ratings
0 2016-01-04   Rainy
1 2016-01-01  Cloudy
2 2016-03-01   Sunny

tx_weather_suff = pd.merge_ordered(austin,houston, on='date', suffixes=['_aus','_hus'])

            date ratings_aus ratings_hus
    0 2016-01-01      Cloudy      Cloudy
    1 2016-01-04         NaN       Rainy
    2 2016-01-17       Sunny         NaN
    3 2016-02-08      Cloudy         NaN
    4 2016-03-01         NaN       Sunny
	
#######################################

# Drop rows with NA - values by subset operation
# Count the number of missing values in each column
print(ri.isnull().sum())

# Drop all rows that are missing 'driver_gender'
ri.dropna(subset=['driver_gender'], inplace=True)

# Count the number of missing values in each column (again)
print(ri.isnull().sum())

# Examine the shape of the DataFrame
print(ri.shape)

#######################################

# Typecast datatype inplace
# Examine the head of the 'is_arrested' column
print(ri.is_arrested.head())

# Change the data type of 'is_arrested' to 'bool'
ri['is_arrested'] = ri.is_arrested.astype('bool')

# Check the data type of 'is_arrested' 
print(ri.is_arrested.dtype)

#######################################

# Typecast to datetime
# Concatenate 'stop_date' and 'stop_time' (separated by a space)
combined = ri.stop_date.str.cat(ri.stop_time,sep=' ')

# Convert 'combined' to datetime format
ri['stop_datetime'] = pd.to_datetime(combined)

#######################################

# Set index
# Set 'stop_datetime' as the index
ri.set_index('stop_datetime', inplace=True)

# Examine the index
print(ri.index)

# Examine the columns
print(ri.columns)

#######################################

# Count the absolute and relative (percentage) of occurences
# Count the unique values in 'violation'
print(ri.violation.value_counts())

# Express the counts as proportions
print(ri.violation.value_counts(normalize=True))

#######################################

# Filter and count proportional occurences
# Create a DataFrame of female drivers
female = ri[ri.driver_gender=='F']

# Create a DataFrame of male drivers
male = ri[ri.driver_gender=='M']

# Compute the violations by female drivers (as proportions)
print(female.violation.value_counts(normalize=True))

# Compute the violations by male drivers (as proportions)
print(male.violation.value_counts(normalize=True))

#######################################

# Calculate appearence
# Create a DataFrame of female drivers stopped for speeding
female_and_speeding = ri[(ri.driver_gender=='F') & (ri.violation=='Speeding') ]

# Compute the stop outcomes for female drivers (as proportions)
print(female_and_speeding.stop_outcome.value_counts(normalize=True) )

    Citation            0.952192
    Warning             0.040074
    Arrest Driver       0.005752
    N/D                 0.000959
    Arrest Passenger    0.000639
    No Action           0.000383
	
#######################################

# Compare dataframes by reindex and dropna
# Import pandas
import pandas as pd

# Reindex names_1981 with index of names_1881: common_names
common_names = names_1981.reindex(names_1881.index)

# Print shape of common_names
print(common_names.shape)

# Drop rows with null counts: common_names
common_names = common_names.dropna()

# Print shape of new common_names
print(common_names.shape)

#################################################

In [2]: weather.iloc[:5,:4]
Out[2]: 
            Max TemperatureF  Mean TemperatureF  Min TemperatureF  Max Dew PointF
Date                                                                             
2013-01-01                32                 28                21              30
2013-01-02                25                 21                17              14
2013-01-03                32                 24                16              19
2013-01-04                30                 28                27              21
2013-01-05                34                 30                25              23

# Extract selected columns from weather as new DataFrame: temps_f
temps_f = weather[['Min TemperatureF', 'Mean TemperatureF', 'Max TemperatureF']]

# Convert temps_f to celsius: temps_c
temps_c = (temps_f-32)*5/9

# Rename 'F' in column names with 'C': temps_c.columns
temps_c.columns = temps_c.columns.str.replace('F','C')

# Print first 5 rows of temps_c
print(temps_c.head())

                Min TemperatureC  Mean TemperatureC  Max TemperatureC
    Date                                                             
    2013-01-01         -6.111111          -2.222222          0.000000
    2013-01-02         -8.333333          -6.111111         -3.888889
    2013-01-03         -8.888889          -4.444444          0.000000
    2013-01-04         -2.777778          -2.222222         -1.111111
    2013-01-05         -3.888889          -1.111111          1.111111
    
#################################################

# Resample dataframe and calculate the rate of change
import pandas as pd

# Read 'GDP.csv' into a DataFrame: gdp
gdp = pd.read_csv('GDP.csv',parse_dates=True, index_col='DATE')

# Slice all the gdp data from 2008 onward: post2008
post2008 = gdp.loc['2008':,:]

# Print the last 8 rows of post2008
print(post2008.tail(8))

# Resample post2008 by year, keeping last(): yearly
yearly = post2008.resample('A').last()

# Print yearly
print(yearly)

# Compute percentage growth of yearly: yearly['growth']
yearly['growth'] = yearly.pct_change()*100

# Print yearly again
print(yearly)

#################################################

# Using the built-in multiply function for row-wise operation
# Import pandas
import pandas as pd

# Read 'sp500.csv' into a DataFrame: sp500
sp500 = pd.read_csv('sp500.csv', parse_dates=True, index_col='Date')

# Read 'exchange.csv' into a DataFrame: exchange
exchange = pd.read_csv('exchange.csv', parse_dates=True, index_col='Date')

# Subset 'Open' & 'Close' columns from sp500: dollars
dollars = sp500[['Open', 'Close']]

# Print the head of dollars
print(dollars.head())

# Convert dollars to pounds: pounds
pounds = dollars.multiply(exchange['GBP/USD'],axis='rows')

# Print the head of pounds
print(pounds.head())

#################################################

# Appending two dataframes (just adding rows)
# Import pandas
import pandas as pd

# Load 'sales-jan-2015.csv' into a DataFrame: jan
jan = pd.read_csv('sales-jan-2015.csv', parse_dates=True, index_col='Date')

# Load 'sales-feb-2015.csv' into a DataFrame: feb
feb = pd.read_csv('sales-feb-2015.csv', parse_dates=True, index_col='Date')

# Load 'sales-mar-2015.csv' into a DataFrame: mar
mar = pd.read_csv('sales-mar-2015.csv', parse_dates=True, index_col='Date')

# Extract the 'Units' column from jan: jan_units
jan_units = jan['Units']

# Extract the 'Units' column from feb: feb_units
feb_units = feb['Units']

# Extract the 'Units' column from mar: mar_units
mar_units = mar['Units']

# Append feb_units and then mar_units to jan_units: quarter1
quarter1 = jan_units.append(feb_units).append(mar_units)

# Print the first slice from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])

# Print the second slice from quarter1
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])

# Compute & print total sales in quarter1
print(quarter1.sum())

#################################################

# The function pd.concat() can concatenate DataFrames horizontally as well as vertically (vertical is the default). To make the DataFrames stack horizontally, you have to specify the keyword argument axis=1 or axis='columns'
# Create a list of weather_max and weather_mean
weather_list = [weather_max, weather_mean]

# Concatenate weather_list horizontally
weather = pd.concat(weather_list, axis=1) # Horizontal stack

# Print weather
print(weather)

#################################################

# Build filenames by iterating and %s operation
#Initialize an empyy list: medals
medals =[]

for medal in medal_types:
    # Create the file name: file_name
    file_name = "%s_top5.csv" % medal
    # Create list of column names: columns
    columns = ['Country', medal]
    # Read file_name into a DataFrame: medal_df
    medal_df = pd.read_csv(file_name,header=0,index_col='Country', names=columns)
    # Append medal_df to medals
    medals.append(medal_df)

# Concatenate medals horizontally: medals_df
medals_df = pd.concat(medals, axis='columns')

# Print medals_df
print(medals_df)

#################################################

# Concatenate by first appending the dataframes. It's a two step mechanisn
for medal in medal_types:

    file_name = "%s_top5.csv" % medal
    
    # Read file_name into a DataFrame: medal_df
    medal_df = pd.read_csv(file_name,index_col='Country')
    
    # Append medal_df to medals
    medals.append(medal_df)
    
# Concatenate medals: medals
medals = pd.concat(medals,keys=['bronze', 'silver', 'gold'])

# Print medals in entirety
print(medals)

#################################################

# Accessing the MultipleIndex by Tupel (,) 
# Create an object to more easily perform multi-index slicing
# Sort the entries of medals: medals_sorted
medals_sorted = medals.sort_index(level=0)
In [4]: medals_sorted.head()
Out[4]: 
                        Total
       Country               
bronze France           475.0
       Germany          454.0
       Soviet Union     584.0
       United Kingdom   505.0
       United States   1052.0
       
# Print the number of Bronze medals won by Germany
print(medals_sorted.loc[('bronze','Germany')])

# Print data about silver medals
print(medals_sorted.loc['silver'])

# Create alias for pd.IndexSlice: idx
idx = pd.IndexSlice

# Print all the data on medals won by the United Kingdom
print(medals_sorted.loc[idx[:,'United Kingdom'],:])

                       Total
       Country              
bronze United Kingdom  505.0
gold   United Kingdom  498.0
silver United Kingdom  591.0

#################################################

# Catentate dataframes horizontally and set the keys to the columns
    # Construct a new DataFrame february with MultiIndexed columns by concatenating the list dataframes.
    # Use axis=1 to stack the DataFrames horizontally and the keyword argument keys=['Hardware', 'Software', 'Service'] to construct a hierarchical Index from each DataFrame.
    # Print summary information from the new DataFrame february using the .info() method. This has been done for you.
    # Create an alias called idx for pd.IndexSlice.
    # Extract a slice called slice_2_8 from february (using .loc[] & idx) that comprises rows between Feb. 2, 2015 to Feb. 8, 2015 from columns under 'Company'.
    # Print the slice_2_8. This has been done for you, so hit 'Submit Answer' to see the sliced data!

# Concatenate dataframes: february
february = pd.concat(dataframes,keys=['Hardware', 'Software', 'Service'], axis=1)

# Print february.info()
print(february.info())

In [5]: february.iloc[:4,:6]
Out[5]: 
                      Hardware                   Software                
                       Company   Product Units    Company   Product Units
Date                                                                     
2015-02-02 08:33:01        NaN       NaN   NaN      Hooli  Software   3.0
2015-02-02 20:54:49  Mediacore  Hardware   9.0        NaN       NaN   NaN
2015-02-03 14:14:18        NaN       NaN   NaN    Initech  Software  13.0
2015-02-04 15:36:29        NaN       NaN   NaN  Streeplex  Software  13.0

<class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 20 entries, 2015-02-02 08:33:01 to 2015-02-26 08:58:51
    Data columns (total 9 columns):
    (Hardware, Company)    5 non-null object
    (Hardware, Product)    5 non-null object
    (Hardware, Units)      5 non-null float64
    (Software, Company)    9 non-null object
    (Software, Product)    9 non-null object
    (Software, Units)      9 non-null float64
    (Service, Company)     6 non-null object
    (Service, Product)     6 non-null object
    (Service, Units)       6 non-null float64
    dtypes: float64(3), object(6)

# Assign pd.IndexSlice: idx
idx = pd.IndexSlice

# Create the slice: slice_2_8
slice_2_8 = february.loc['2015-Feb-02':'2015-Feb-08', idx[:, 'Company']]

# Print slice_2_8
print(slice_2_8)

#################################################

In [1]: jan.head()
Out[1]: 
                  Date    Company   Product  Units
0  2015-01-21 19:13:21  Streeplex  Hardware     11
1  2015-01-09 05:23:51  Streeplex   Service      8
2  2015-01-06 17:19:34    Initech  Hardware     17
3  2015-01-02 09:51:06      Hooli  Hardware     16
4  2015-01-11 14:51:02      Hooli  Hardware     11

# Make the list of tuples: month_list
month_list = [('january', jan), ('february', feb), ('march', mar)]

# Create an empty dictionary: month_dict
month_dict = {}

for month_name, month_data in month_list:

    # Group month_data: month_dict[month_name]
    month_dict[month_name] = month_data.groupby('Company').sum()

# Concatenate data in month_dict: sales
sales = pd.concat(month_dict)

# Print sales
print(sales)

<script.py> output:
                              Units
             Company               
    february Acme Coporation     34
             Hooli               30
             Initech             30
             Mediacore           45
             Streeplex           37
    january  Acme Coporation     76
             Hooli               70
             Initech             37
             Mediacore           15
             Streeplex           50
    march    Acme Coporation      5
             Hooli               37
             Initech             68
             Mediacore           68
             Streeplex           40

# Print all sales by Mediacore
idx = pd.IndexSlice
print(sales.loc[idx[:, 'Mediacore'], :])

                        Units
             Company         
    february Mediacore     45
    january  Mediacore     15
    march    Mediacore     68
    

#################################################

# When concatenating dataframes horizontally, it makes sense to set the keys accordingly

In [1]: bronze
Out[1]: 
                 Total
Country               
United States   1052.0
Soviet Union     584.0
United Kingdom   505.0
France           475.0
Germany          454.0

In [2]: silver
Out[2]: 
                 Total
Country               
United States   1195.0
Soviet Union     627.0
United Kingdom   591.0
France           461.0
Italy            394.0

# Create the list of DataFrames: medal_list
medal_list = [bronze,silver,gold]

# Concatenate medal_list horizontally using an inner join: medals
medals = pd.concat(medal_list,keys=['bronze', 'silver', 'gold'], axis=1, join='inner')

# Print medals
print(medals)


<script.py> output:
                    bronze  silver    gold
                     Total   Total   Total
    Country                               
    United States   1052.0  1195.0  2088.0
    Soviet Union     584.0   627.0   838.0
    United Kingdom   505.0   591.0   498.0
    
#################################################

# Comprehension of 10 Years GDP by usage of resample 
    # Make a new DataFrame china_annual by resampling the DataFrame china with .resample('A').last() (i.e., with annual frequency) and chaining two method calls:
    # Chain .pct_change(10) as an aggregation method to compute the percentage change with an offset of ten years.
    # Chain .dropna() to eliminate rows containing null values.
    # Make a new DataFrame us_annual by resampling the DataFrame us exactly as you resampled china.
    # Concatenate china_annual and us_annual to construct a DataFrame called gdp. Use join='inner' to perform an inner join and use axis=1 to concatenate horizontally.
    # Print the result of resampling gdp every decade (i.e., using .resample('10A')) and aggregating with the method .last(). This has been done for you, so hit 'Submit Answer' to see the result!

# Resample and tidy china: china_annual
china_annual = china.resample('A').last().pct_change(10).dropna()

# Resample and tidy us: us_annual
us_annual = us.resample('A').last().pct_change(10).dropna()

# Concatenate china_annual and us_annual: gdp
gdp = pd.concat([china_annual,us_annual], join='inner', axis=1)

# Resample gdp and print
print(gdp.resample('10A').last())

<script.py> output:
                   China        US
    Year                          
    1971-12-31  0.988860  1.052270
    1981-12-31  0.972048  1.750922
    1991-12-31  0.962528  0.912380
    2001-12-31  2.492511  0.704219
    2011-12-31  4.623958  0.475082
    2021-12-31  3.789936  0.361780

#################################################

# merge covers more cases in catenating dataframes
combined = pd.merge(revenue, managers, on='city')

#################################################

In [1]: revenue
          city  branch_id state  revenue
0       Austin         10    TX      100
1       Denver         20    CO       83
2  Springfield         30    IL        4
3    Mendocino         47    CA      200

In [2]: managers
        branch  branch_id state   manager
0       Austin         10    TX  Charlers
1       Denver         20    CO      Joel
2    Mendocino         47    CA     Brett
3  Springfield         31    MO     Sally

# Merge revenue & managers on 'city' & 'branch': combined
combined = pd.merge(revenue, managers, left_on='city', right_on='branch')
        
# Print combined
print(combined)
          city  branch_id_x state_x  revenue       branch  branch_id_y state_y   manager
0       Austin           10      TX      100       Austin           10      TX  Charlers
1       Denver           20      CO       83       Denver           20      CO      Joel
2  Springfield           30      IL        4  Springfield           31      MO     Sally
3    Mendocino           47      CA      200    Mendocino           47      CA     Brett

#################################################

# The pivot_table() method has 4 arguments
## index: Index of pivot table
## values: columns to aggregate
## aggfunc: function to apply for aggregation
## columns: Categories

In [2]: medals.head()
              Athlete  NOC   Medal  Edition
0       HAJOS, Alfred  HUN    Gold     1896
1    HERSCHMANN, Otto  AUT  Silver     1896
2   DRIVAS, Dimitrios  GRE  Bronze     1896
3  MALOKINIS, Ioannis  GRE    Gold     1896
4  CHASAPIS, Spiridon  GRE  Silver     1896

medals.pivot_table(index='Edition',values='Athlete',columns='NOC', aggfunc='count')

    NOC      AFG  AHO  ALG   ANZ  ARG  ...  VIE  YUG  ZAM  ZIM   ZZX
    Edition                            ...                          
    1896     NaN  NaN  NaN   NaN  NaN  ...  NaN  NaN  NaN  NaN   6.0
    1900     NaN  NaN  NaN   NaN  NaN  ...  NaN  NaN  NaN  NaN  34.0
    1904     NaN  NaN  NaN   NaN  NaN  ...  NaN  NaN  NaN  NaN   8.0
    1908     NaN  NaN  NaN  19.0  NaN  ...  NaN  NaN  NaN  NaN   NaN
    1912     NaN  NaN  NaN  10.0  NaN  ...  NaN  NaN  NaN  NaN   NaN

#################################################

# Check the data type of 'search_conducted'
print(ri.search_conducted.dtype)

# Calculate the search rate by counting the values
print(ri.search_conducted.value_counts(normalize=True))

# Calculate the search rate by taking the mean
print(ri.search_conducted.mean())

#################################################

# Calculate mean() over group
# Calculate the search rate for both groups simultaneously
print(ri.groupby('driver_gender').search_conducted.mean())

#################################################

# Using groupby on sub-categories
# Reverse the ordering to group by violation before gender
print(ri.groupby(['violation','driver_gender']).search_conducted.mean())

#################################################

# Produce a summary of occurences and create a boolean mask for a special occurence
# Count the 'search_type' values
print(ri.search_type.value_counts() )

# Check if 'search_type' contains the string 'Protective Frisk'
ri['frisk'] = ri.search_type.str.contains('Protective Frisk', na=False)

# Check the data type of 'frisk'
print(ri.frisk.dtype )

# Take the sum of 'frisk'
print(ri.frisk.sum() )

#################################################

# Bootstrap mass replicate function
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data,func)

    return bs_replicates

#################################################

# calculate standard deviation by bootstrap without derivation
# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(data=rainfall, func=np.mean, size=10000)

# Compute and print SEM
sem = np.std(rainfall) / np.sqrt(len(rainfall))
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

#################################################

# Bootstrap of pairs and apply linear regression -> yield m,b

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x,bs_y,1)

    return bs_slope_reps, bs_intercept_reps

#################################################

# Generate array of x-values for bootstrap lines: x
x = np.array([0,100])

# Plot the bootstrap lines
# This actually works because a line consists of two points x=0 and x=100
for i in range(100):
    _ = plt.plot(x, bs_slope_reps[i]*x + bs_intercept_reps[i],linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(illiteracy, fertility, marker='.',linestyle='none' )

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()

#################################################

Bootstrap Test v.s. Permutationstest
Permutationstest:
	H0: immer Gleichheit zweier Verteilungen
	bedingter Test
	exakt, wenn Austauschbarkeit besteht
Boostrap Test:
	H0: allgemeiner (Mittelwert, Median)
	kein bedingter Test
	nicht exakt

#################################################

# Permutationstest Sampling of two datasets
def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2
	
#################################################

# Test of identical distribution
for _ in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(rain_june, rain_november)

    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1,y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(rain_june)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()

#################################################

# Permutation replicate function computes an array of statistical solutions
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2) # any custom function

    return perm_replicates
	
#################################################

# Comparing the force push of two different frog types
# Pipeline for hypothesis testing
# 1- Clearly state the null hypothesis
# 2- Define your test statistic
# 3- Generate many sets of simulated data assuming the null hypothesis is true
# 4- Compute the test statistic for each simulated data set
# 5- The p-value is the fraction of your simulated data sets for
	# which the test statistic is at least as extreme as for the real data

# P-Value calculation
def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1)-np.mean(data_2)

    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means.
# H0: Would we get the difference of means 
empirical_diff_means = diff_of_means(force_a, force_b)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(force_a, force_b, diff_of_means , size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

# Print the result
print('p-value =', p)

# The p-value tells you that there is about a 0.6% chance that you would get the difference
# of means observed in the experiment if frogs were exactly the same.
# A p-value below 0.01 is typically said to be "statistically significant,"

#################################################

