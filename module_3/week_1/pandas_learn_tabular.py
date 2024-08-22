import pandas as pd

pd.options.mode.copy_on_write = True

dataset_path = "IMDB-Movie-Data.csv"
REVENUE_INDEX = 'Revenue (Millions)'
# 1-Read data
data = pd.read_csv(dataset_path)
data_indexed = pd.read_csv(dataset_path, index_col="Title")

# 3-View the data
print(data.head())

# 4-Understand some basic information about the data:
data.info()
print(data.describe())

# Data selection - Indexing and Slicing data
# Extract data as series
genre = data['Genre']
print(genre)

# Extract data as dataframe
genre = data[['Genre']]
print(genre)

# Combine multi column to new dataframe
some_cols = data[['Title', 'Genre', 'Actors', 'Director', 'Rating']]
print(some_cols)

# Slicing get specific range of row
print(data.iloc[10:15][['Title', 'Rating', REVENUE_INDEX]])

# 5-Data Selection - Based on Conditional filtering
movie = data[((data['Year'] >= 2010) & (data['Year'] <= 2015))
             & (data['Rating'] < 6.0)
             & (data[REVENUE_INDEX] > data[REVENUE_INDEX].quantile(0.95))]
print(movie)

# 6-Groupby Operation
print(data.groupby('Director')[['Rating']].mean().head())

# 7-Sorting Operation
print(data.groupby('Director')[['Rating']].mean().sort_values(['Rating'], ascending=False).head())
print("")

# 8-View missing values
print(data.isnull().sum())

# 9-Deal with missing values-Deleting
# If we really want to delete add inplace = True
print(data.drop('Metascore', axis=1).head())

# 10-Deal with missing values-Filling
revenue_mean = data_indexed[REVENUE_INDEX].mean()
print("The mean revenue is: ", revenue_mean)

a = data_indexed[REVENUE_INDEX].fillna(revenue_mean)


# 11-Apply function
def rating_group(rating):
    if rating >= 7.5:
        return "Good"
    elif rating >= 6.0:
        return "Average"
    else:
        return "Bad"


data['Rating_category'] = data['Rating'].apply(rating_group)
print(data[['Title', 'Director', 'Rating', 'Rating_category']].head(5))
