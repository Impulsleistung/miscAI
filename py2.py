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

