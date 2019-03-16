import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading books file
books = pd.read_csv('BX-Books.csv', sep=';', 
	encoding ='latin-1', error_bad_lines=False)
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 
'yearOfPublication', 
'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

# reading users file
users = pd.read_csv('BX-Users.csv', sep=';', 
	encoding ='latin-1', error_bad_lines=False)
users.columns = ['userID', 'Location', 'Age']

# reading ratings file
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', 
	encoding ='latin-1', error_bad_lines=False)
ratings.columns = ['userID', 'ISBN', 'bookRating']

# printing details about books, users and ratings
print(books.shape)
print(users.shape)
print(ratings.shape)

# recommendation according to number of ratings
rating_count = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
rating_count.sort_values('bookRating', ascending=False).head()

# connecting file rating and books so we can see the actual most rated books names
most_rated_books = pd.DataFrame(['0971880107', '0316666343', 
	'0385504209', '0060928336', '0312195516'], index=np.arange(5), columns = ['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
# print
most_rated_books_summary

# finding the average rating and the number of ratings each book recieved
average_rating = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].mean())
average_rating['ratingCount'] = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
# print
print(average_rating.sort_values('ratingCount', ascending=False).head())


# books with best average ratings with condition that the book has to has more than 50 ratings
average_rating1 = average_rating[average_rating['ratingCount']>= 50]
best_average_rating = average_rating1.sort_values('bookRating', ascending=False).head(10)
best_average_rated_books = pd.merge(best_average_rating, books, on = 'ISBN')
print(best_average_rated_books)


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
books_corr_to_bones = pd.DataFrame(['0312291639', '0316601950', '0446610038', '0446672211', '0385265700', 
	'0345342968', '0060930535', '0375707972', '0684872153'], index =np.arange(9),
	columns = ['ISBN'])
corr_books = pd.merge(books_corr_to_bones, books, on = 'ISBN')
print(corr_books)




