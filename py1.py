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

