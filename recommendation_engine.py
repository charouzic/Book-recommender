import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading books file
books = pd.read_csv('BX-Books.csv', sep=';', 
	encoding ='latin-1', error_bad_lines=False)
books.columns = ['ISBN', 'bookTitle', 
'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

# reading users file
users = pd.read_csv('BX-Users.csv', sep=';', encoding ='latin-1', error_bad_lines=False)
users.columns = ['userID', 'Location', 'Age']

# reading ratings file
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding ='latin-1', error_bad_lines=False)
ratings.columns = ['userID', 'ISBN', 'bookRating']

# printing details about books, users and ratings
print(books.shape)
print(users.shape)
print(ratings.shape)

# overview of ratings
plt.rc("font", size=15)
ratings.bookRating.value_counts(sort=False).plot(kind='bar')
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('system1.png', bbox_inches='tight')
plt.show()
# most of the reviews are accumulated at 0

# histogram of users' age
users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
plt.title('Age Distribution\n')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('system2.png', bbox_inches='tight')
plt.show()
# biggest group of users is between 20 and 40

# recommendation according to number of ratings
rating_count = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
rating_count.sort_values('bookRating', ascending=False).head()

# connecting file rating and books so we can see the actual most rated books' names
most_rated_books = pd.DataFrame(['0971880107', '0316666343', 
	'0385504209', '0060928336', '0312195516'], index=np.arange(5), columns = ['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
# print
most_rated_books_summary

# finding the average rating and the number of ratings each book recieved
average_rating = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].mean())
average_rating['ratingCount'] = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
print(average_rating.sort_values('ratingCount', ascending=False).head())


# 10 books with best average rating under condition that the book has to has more than 50 ratings
average_rating1 = average_rating[average_rating['ratingCount']>= 29]
best_average_rating = average_rating1.sort_values('bookRating', ascending=False).head(10)
best_average_rated_books = pd.merge(best_average_rating, books, on = 'ISBN')
#print
(best_average_rated_books)


# excluding users with less than 250 ratings and books with less than 150 ratings for statistical significance
counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1>= 250].index)]
counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 150].index)]


# converting table to 2D matrix (sparse/řídká/ data)
ratings_pivot = ratings.pivot(index='userID', columns ='ISBN').bookRating
userID = ratings_pivot.index
ISBN = ratings_pivot.columns
print(ratings_pivot.shape)
# print
ratings_pivot.head()


# finding books that are best correlated with the 2nd most rated book ("The Kovely Bones: A Novel")
wanted_book = '0316666343'
bones_ratings = ratings_pivot[wanted_book]
similar_to_bones = ratings_pivot.corrwith(bones_ratings)
corr_bones = pd.DataFrame(similar_to_bones, columns=['pearsonR'])
corr_bones.dropna(inplace=True)
corr_summary = corr_bones.join(average_rating['ratingCount'])
# print
corr_summary[corr_summary['ratingCount']>= 300].sort_values('pearsonR', ascending=False).head(10)

# we need book names not ISBN -> merging
books_corr_to_bones = pd.DataFrame(['0312291639', '0316601950', '0446610038', '0446672211', 
	'0385265700', '0345342968', '0060930535', '0375707972', '0684872153'], index =np.arange(9),
	columns = ['ISBN'])
corr_books = pd.merge(books_corr_to_bones, books, on = 'ISBN')
print(corr_books)


### collaborative filtering -> kNN (k-Nearest Neighbors) ###
""" algorithm based on common book ratings"""

# combining books data and ratings data
combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis = 1)
print(combine_book_rating.head())

# grouping books by titles + creating new column
combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])

book_ratingCount = (combine_book_rating.groupby(by = ['bookTitle'])['bookRating']. count().
	reset_index().
	rename(columns = {'bookRating': 'totalRatingCount'})
	[['bookTitle', 'totalRatingCount']]
	)
#print
(book_ratingCount.head())

# comparing rating data with the total rating count
rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')
print(rating_with_totalRatingCount.head())

# statistics of total rating count
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#print
(book_ratingCount['totalRatingCount'].describe())
# median book has been rated only once

# top of the distribution
#print
(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))
# about 1% of the books recieved 29 or more ratings

# getting unique books
popularity_threshold = 29
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
#print
(rating_popular_book.head())

# improving computing speed -> only users from US and Canada
combined = rating_popular_book.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')

us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating = us_canada_user_rating.drop('Age', axis = 1)
#print
(us_canada_user_rating.head())

#implementing kNN
""" converting the table to 2D matrix and filling the missing values with 0"""
from scipy.sparse import csr_matrix

us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot( index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(us_canada_user_rating_matrix)


# testing our model and getting some recommendation
query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
	if i == 0:
		print("Recommendations of {0}:\n".format(us_canada_user_rating_pivot.index[query_index]))
	else:
		print("{0}: {1}, with distance of {2}:".format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))


### collaborative filtering using matrix factorization ###
us_canada_user_rating_pivot2 = us_canada_user_rating.pivot(index = 'userID', columns = 'bookTitle', values = 'bookRating').fillna(0)
#print
(us_canada_user_rating_pivot2.head())
#print
(us_canada_user_rating_pivot2.shape)

X = us_canada_user_rating_pivot2.values.T

import sklearn
from sklearn.decomposition import TruncatedSVD as SVD

SVD = SVD(n_components =12, random_state = 17)
matrix = SVD.fit_transform(X)
#print
(matrix.shape)

import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)
corr = np.corrcoef(matrix)
#print
(corr.shape)

us_canada_book_title = us_canada_user_rating_pivot2.columns
us_canada_book_list = list(us_canada_book_title)
coffey_hands = us_canada_book_list.index("The Green Mile: Coffey's Hands (Green Mile Series)")
#print
(coffey_hands)

corr_coffey_hands = corr[coffey_hands]
#print
(list(us_canada_book_title[(corr_coffey_hands >= 0.9)]))
