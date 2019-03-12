import pandas as pd
import numpy as np


data = pd.read_csv("BX-Book-Ratings.csv", error_bad_lines=False, 
	names = ['User-ID', 'ISBN', 'Book-Rating'])

data.head()

